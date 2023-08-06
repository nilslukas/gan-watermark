import os
import time
from copy import deepcopy
from os.path import dirname
from typing import Union

import wandb
import torch
import torchvision
from accelerate import Accelerator
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.criteria.id_loss import IDLoss
from src.criteria.lpips_loss import LPIPSLoss
from src.model_converter.mappers.layer_wm_mapper import WMMapperGroup
from src.model_converter.model_converter import ModelConverter
from src.models.gan import GAN
from src.models.model_factory import ModelFactory
from src.utils.highlited_print import bcolors, print_dict_highlighted
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import compute_bitwise_acc, plot_images
from src.watermarking_key.wm_key import WatermarkingKey


class PTWWatermarkingKey(WatermarkingKey):
    MODEL_STATE_DICT_KEY = "ptw_state_dict"  # the key in the dictionary to save the model state

    def __init__(self, wm_key_args: WatermarkingKeyArgs, **kwargs):
        super().__init__(wm_key_args=wm_key_args, **kwargs)

        print(
            f"> (PTW) Instantiating {bcolors.OKCYAN}{wm_key_args.decoder_arch}{bcolors.ENDC} with {wm_key_args.bitlen} bits.")
        self.decoder = ModelFactory.load_decoder(wm_key_args.bitlen, decoder_arch=wm_key_args.decoder_arch)
        self.decoder.to(self.env_args.device)

        self.mappers: Union[None, WMMapperGroup] = None
        self.logging_data: dict = {}  # log during training

        if os.path.exists(self.wm_key_args.key_ckpt):  # restore state
            self.load(self.wm_key_args.key_ckpt)

    def extract(self, x: torch.Tensor, sigmoid=True, **kwargs) -> torch.Tensor:
        """ Extract a watermarking embedded_message from an image tensor. """
        return torch.sigmoid(self.decoder(x)) if sigmoid else self.decoder(x)

    def load(self, ckpt_fn: str = None) -> None:
        """ Load the state from a checkpoint. """
        ckpt_fn = self.wm_key_args.key_ckpt if ckpt_fn is None else ckpt_fn
        data = torch.load(ckpt_fn)
        self.load_state_dict(data[self.MODEL_STATE_DICT_KEY])
        print(f"> Restored the state from {bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}. ")

    def save(self, ckpt_fn: str = None) -> dict:
        """ Create a single '*.pt' file. Returns only the dict if no ckpt_fn is provided.  """
        save_dict = {
            self.MODEL_STATE_DICT_KEY: self.state_dict(),
            **super().save()
        }
        if ckpt_fn is not None:
            print(f"> Writing a PTW decoder to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'.")
            torch.save(save_dict, ckpt_fn)
        return save_dict

    def initialize_training(self):
        os.makedirs(dirname(self.wm_key_args.key_ckpt), exist_ok=True)
        print()
        print(f"> --------------------------------------------- ")
        print(f"> Initiating training of a PTW watermarking key.")
        print(f"> Logging: {self.env_args.logging_tool}")
        print(
            f"> Checkpoint directory: '{bcolors.OKGREEN}{os.path.abspath(self.wm_key_args.key_ckpt)}{bcolors.ENDC}'")
        print(f"> Bit length: {self.wm_key_args.bitlen}")
        print(f"> --------------------------------------------- ")
        print()

    def setup_logging(self):
        # Set up the logging tool.
        run = None
        if self.env_args.logging_tool == "wandb":
            wandb.login()
            run = wandb.init(project="ptw_keygen")
        return run

    def prepare_generators_and_optimizers(self, g_target):
        # Prepare the generators.
        g_frozen = deepcopy(g_target)

        print(self.wm_key_args)

        model_converter = ModelConverter(self.wm_key_args)
        g_target, self.mappers = model_converter.convert(g_target)
        assert self.mappers.size() > 0, "Need to specify at least one mapper to embed a watermark!"

        # Prepare optimizers.
        all_mapper_params = []
        for mapper in self.mappers:
            all_mapper_params += list(mapper.parameters())
            mapper.train().requires_grad_(True)
        g_target.activate_gradients()

        opt_decoder = Adam(self.decoder.parameters(), lr=self.wm_key_args.lr_decoder)
        opt_mapper = Adam(all_mapper_params, lr=self.wm_key_args.lr_mapper)
        return g_frozen, opt_decoder, opt_mapper

    def prepare_losses(self) -> dict:
        return {
            "bce": BCEWithLogitsLoss(),  # classification loss to extract the watermark
            "lpips": LPIPSLoss(),  # LPIPS loss between outputs of pivot and trainable
            "id": IDLoss(self.wm_key_args.ir_se50_weights)  # optional for facial data.
        }

    def prepare_processing_and_logging(self):
        self.logging_data['log'] = {}
        self.logging_data['best_bit_acc'] = 0.0

        # Experimental & Optional: Increase robustness by adding perturbations. Adapt to attacks.
        preprocessing = transforms.Compose([])
        if self.wm_key_args.add_preprocessing:
            preprocessing = transforms.Compose([
                transforms.RandomErasing(),
                transforms.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.95, 1.0))
            ])
        return preprocessing

    def learn(self,
              g_target: GAN,
              **kwargs):
        """ Train a PTW watermarking key from scratch with a vanilla, pre-trained generator.
            We replace layers in the given generator to accept watermarking messages.
        """
        assert self.wm_key_args.key_ckpt is not None, "Please specify a key_ckpt to save the key to."
        print(self.env_args)

        self.initialize_training()
        run = self.setup_logging()
        g_frozen, opt_decoder, opt_mapper = self.prepare_generators_and_optimizers(g_target)
        losses_funcs = self.prepare_losses()
        preprocessing = self.prepare_processing_and_logging()

        accelerator = Accelerator()
        g_frozen, g_target, self.mappers, opt_decoder, opt_mapper = accelerator.prepare(g_frozen,
                                                                                        g_target,
                                                                                        self.mappers,
                                                                                        opt_decoder,
                                                                                        opt_mapper)

        # The training loop (infinity)
        bit_acc = SmoothedValue()  # a smoothed representation of the bit accuracy.
        best_bit_acc = 0
        step = 0
        start_time = time.time()
        print(f"> Initiating PTW training ..")

        with tqdm(total=50, desc="Training PTW Key (infinity loop)") as pbar:
            while True:
                loss_dict = {"iter": step}

                g_target.G.eval()
                g_frozen.G.eval()
                self.decoder.train()

                # Collect reference rollouts.
                with torch.no_grad():
                    w, x = g_frozen.generate(batch_size=self.env_args.batch_size,
                                             truncation_psi=self.wm_key_args.truncation_psi)
                    msg = self.gen_msg(n=x.shape[0]).to(self.env_args.device).float()

                opt_mapper.zero_grad()
                opt_decoder.zero_grad()

                self.mappers.set_msg(msg)  # forward the embedded_message to the mappers.
                _, x_wm = g_target.generate(w=w)
                extracted_msg = self.extract(preprocessing(x_wm), sigmoid=False)

                # classifier loss
                loss_bce = losses_funcs['bce'](extracted_msg, msg)
                loss_dict['loss_bce'] = float(loss_bce)
                loss = loss_bce

                if self.wm_key_args.keygen_lambda_lpips > 0:
                    loss_lpips = self.wm_key_args.keygen_lambda_lpips * losses_funcs['lpips'](x, x_wm).mean()
                    loss_dict['loss_lpips'] = float(loss_lpips)
                    loss += loss_lpips

                if self.wm_key_args.keygen_lambda_id > 0:
                    loss_id = self.wm_key_args.keygen_lambda_id * losses_funcs['id'](x_wm, x)[0].mean()
                    loss_dict['loss_id'] = float(loss_id)
                    loss += loss_id

                accelerator.backward(loss)

                if (step + 1) % self.env_args.gradient_accumulation_steps == 0:
                    opt_decoder.step()
                    opt_mapper.step()

                # Logging
                bit_acc.update(compute_bitwise_acc(msg, torch.sigmoid(extracted_msg)))
                loss_dict['bit_acc'] = bit_acc.avg
                loss_dict['capacity'] = max(0, 2 * (
                        (bit_acc.avg / 100) * self.wm_key_args.bitlen - 0.5 * self.wm_key_args.bitlen))

                if step % self.env_args.log_every == 0:
                    print()
                    print(f"----------------------------------")
                    print(f"> Time Elapsed: {time.time() - start_time:.2f}s")
                    print(
                        f"> Step={step}, Bits={loss_dict['capacity']:.2f}, Bit_acc={bit_acc.avg:.2f}%, Total Bits: {self.wm_key_args.bitlen}")
                    print(
                        f"> Save to ckpt: '{bcolors.OKGREEN}{os.path.abspath(self.wm_key_args.key_ckpt)}{bcolors.ENDC}'")
                    print(f"----------------------------------")
                    print()
                    pbar.reset(total=self.env_args.log_every)

                ## LOGGING
                if step % self.env_args.log_every == 0:
                    top = [x for x in x_wm[:3]]
                    middle = [x for x in x[:3]]
                    bottom = [preprocessing(x - y) for x, y in zip(x_wm[:3], x[:3])]

                    plot_images(torch.stack(top + middle + bottom, 0), n_row=3,
                                title=f"step={step}, bits={loss_dict['capacity']:.2f}")
                    if accelerator.is_local_main_process and run is not None:
                        images = wandb.Image(
                            torchvision.utils.make_grid(torch.stack(top + middle + bottom, 0),
                                                        nrow=3, range=(-1, 1), scale_each=True,
                                                        normalize=True),
                            caption="Top: Watermarked, Middle: Original, Bottom: Diff"
                        )
                        wandb.log({"examples": images})

                if accelerator.is_local_main_process and run is not None:
                    wandb.log({**loss_dict})  # log to wandb

                ## EVALUATION
                if step % self.env_args.save_every == 0 and step > 0:
                    with torch.no_grad():
                        self.decoder.eval()
                        msg = self.gen_msg(n=self.env_args.eval_batch_size).to(self.env_args.device)
                        self.mappers.set_msg(msg)  # forward the embedded_message to the mappers
                        _, x = g_target.generate(batch_size=self.env_args.eval_batch_size,
                                                 truncation_psi=self.wm_key_args.truncation_psi)
                        # bit acc
                        msg_pred = self.extract(x, sigmoid=True)
                        cur_bit_acc = compute_bitwise_acc(msg, msg_pred)
                        is_better = cur_bit_acc > best_bit_acc
                        print()
                        print(f"----------------------------------")
                        print(f"> TESTING PHASE ")
                        print(
                            f"> Evaluate bit acc: {cur_bit_acc:.2f}% {bcolors.OKGREEN}(best: {best_bit_acc:.2f}%){bcolors.ENDC}")
                        if is_better:
                            print(f"> Improvement detected! Saving model .. ")
                            best_bit_acc = cur_bit_acc
                            ckpt_fn = self.wm_key_args.key_ckpt
                            self.save(ckpt_fn)
                        print(f"----------------------------------")
                step += 1
                pbar.update(1)
                pbar.set_description(f"Training PTW Key (infinity loop, Acc: {bit_acc.avg:.2f}")
