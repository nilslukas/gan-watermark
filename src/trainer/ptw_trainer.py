import os
from copy import deepcopy
import time
import wandb

import torch
import torchvision
from accelerate import Accelerator
from torch.optim import Adam
from tqdm import tqdm

from src.arguments.embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs
from src.criteria.id_loss import IDLoss
from src.criteria.lpips_loss import LPIPSLoss
from src.models.gan import GAN
from src.trainer.trainer import Trainer
from src.utils.highlited_print import bcolors, print_warning
from src.utils.smoothed_value import SmoothedValue
from src.utils.utils import compute_bitwise_acc, plot_images
from src.watermarking_key.wm_key import WatermarkingKey


class PTWTrainer(Trainer):

    EMBED_ARGS_KEY: str = "embed_args"
    WM_KEY_KEY: str = "wm_key_key"

    def __init__(self, embed_args: EmbedArgs, env_args: EnvArgs):
        self.embed_args = embed_args
        self.env_args = env_args
        self.generator: GAN = None
        self.wm_key: WatermarkingKey = None

    def save(self, ckpt: str):
        if ckpt is None:
            print_warning(f"> No checkpoint path was given. Skipping saving the model.")
            return
        torch.save({
            **self.generator.save(),
            **self.wm_key.save(),
            self.EMBED_ARGS_KEY: self.embed_args
        }, ckpt)
        print(f"> Saved watermarked model to '{bcolors.OKGREEN}{os.path.abspath(ckpt)}{bcolors.ENDC}'.")

    def setup_logging(self):
        # Set up the logging tool.
        run = None
        if self.env_args.logging_tool == "wandb":
            wandb.login()
            run = wandb.init(project="ptw_embed")
        return run

    def train(self,
              generator: GAN,  # the generator that should be watermarked with a embedded_message.
              wm_key: WatermarkingKey):
        """ Embed the watermark into a generator.
        """
        msg = WatermarkingKey.str_to_bits(wm_key.wm_key_args.message).unsqueeze(0)[:, :wm_key.wm_key_args.bitlen]
        msg = msg.repeat([self.env_args.batch_size, 1]).to(self.env_args.device)

        lpips_loss = LPIPSLoss()
        id_loss = IDLoss(ir_se50_weights=wm_key.wm_key_args.ir_se50_weights)

        reference_generator = deepcopy(generator)
        generator.activate_gradients()
        opt = Adam(generator.G.synthesis.parameters(),  # only the synthesis layers.
                   lr=self.embed_args.ptw_lr,
                   betas=(0 ** (4 / 5), 0.99 ** (4 / 5)))

        accelerator = Accelerator()
        generator, reference_generator, opt = accelerator.prepare(generator, reference_generator, opt)
        self.generator = generator
        self.wm_key = wm_key

        bit_acc = SmoothedValue()
        step = 0
        run = self.setup_logging()
        start_time = time.time()
        with tqdm(total=self.env_args.log_every, desc="Training PTW Key (infinity loop)") as pbar:
            while True:  # infinity loop
                opt.zero_grad()
                wm_key.eval()
                generator.G.train()

                with torch.no_grad():
                    w_frozen, x_frozen = reference_generator.generate(batch_size=self.env_args.batch_size,
                                                                      truncation_psi=wm_key.wm_key_args.truncation_psi)
                _, x_train = generator.generate(w=w_frozen)

                # Compute losses.
                loss, loss_dict = 0, {}

                loss_watermark = wm_key.loss(x_train, msg)
                loss_dict["loss_watermark"] = loss_watermark
                loss += loss_watermark

                loss_lpips = self.embed_args.lambda_lpips * lpips_loss(x_train, x_frozen).mean()
                loss_dict["loss_lpips"] = loss_lpips
                loss += loss_lpips

                if self.embed_args.lambda_id > 0:
                    loss_id = self.embed_args.lambda_id * id_loss(x_train, x_frozen)[0].mean()
                    loss_dict["loss_id"] = loss_id
                    loss += loss_id

                accelerator.backward(loss)
                if (step + 1) % self.env_args.gradient_accumulation_steps == 0:
                    opt.step()

                bit_acc.update(wm_key.validate(x_train, msg))
                loss_dict['capacity'] = max(0,
                                       2. * (bit_acc.avg * wm_key.wm_key_args.bitlen - 0.5 * wm_key.wm_key_args.bitlen))
                loss_dict["bit_acc"] = bit_acc.avg

                ## Logging
                if step % self.env_args.log_every == 0:
                    print()
                    print(f"----------------------------------------------------------")
                    print(f"> Time Elapsed: {time.time() - start_time:.2f}s")
                    print(
                        f"> Step={step}, Bits={loss_dict['capacity']:.2f}, Bit_acc={bit_acc.avg*100:.2f}%, Total Bits: {wm_key.wm_key_args.bitlen}")
                    print(f"> Save to: '{bcolors.OKGREEN}{self.embed_args.ckpt}{bcolors.ENDC}'")
                    print(f"----------------------------------------------------------")
                    print()
                    pbar.reset(total=self.env_args.log_every)

                ## LOGGING
                if step % self.env_args.log_every == 0:
                    top = [x for x in x_train[:3]]
                    middle = [x for x in x_frozen[:3]]
                    bottom = [x - y for x, y in zip(x_train[:3], x_frozen[:3])]

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
                        _, x = generator.generate(batch_size=self.env_args.eval_batch_size,
                                                  truncation_psi=wm_key.wm_key_args.truncation_psi)
                        msg_eval = msg[0].unsqueeze(0).repeat([self.env_args.eval_batch_size, 1]).to(
                            self.env_args.device)
                        # bit acc
                        msg_pred = wm_key.extract(x, sigmoid=True)
                        cur_bit_acc = compute_bitwise_acc(msg_eval, msg_pred)
                        print()
                        print(f"----------------------------------------------------------")
                        print(f"> TESTING PHASE ")
                        print(
                            f"> Evaluate bit acc: {cur_bit_acc:.2f}% {bcolors.OKGREEN}(best: {cur_bit_acc:.2f}%){bcolors.ENDC}")
                        ckpt_fn = self.embed_args.ckpt
                        self.save(ckpt_fn)
                        print(f"----------------------------------------------------------")
                step += 1
                pbar.update(1)
                pbar.set_description(f"Pivotal Tuning (infinity loop), Acc: {100*bit_acc.avg:.2f}%")
