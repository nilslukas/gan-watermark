import copy
from typing import Callable

import numpy as np
import torch

import src
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.compatibility.stylegan_xl import legacy, dnnlib
from src.models.gan import GAN
from src.utils.highlited_print import bcolors


class StyleGAN(GAN):
    """ A wrapper for the StyleGAN2 and StyleGAN3 classes.
    """

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        super().__init__(model_args=model_args)
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.device = self.env_args.device
        self.ckpt = None
        self.post_processing_hooks = []

    def add_post_processing(self, post_processing_fn: Callable):
        last_layer = None
        for name, layer in self.G.synthesis.named_children():
            if "synthesis" in str(layer).lower():
                last_layer = layer
            for layer in layer.children():
                if "synthesis" in str(layer).lower() or "torgb" in str(layer).lower():
                    last_layer = layer
        self.post_processing_hooks.append(last_layer.register_forward_hook(
            lambda layer, _, output: post_processing_fn(output)
        ))

    def clear_post_processing(self):
        for hook in self.post_processing_hooks:
            hook.remove()
        self.post_processing_hooks = []

    def get_discriminator(self):
        return self.D

    def load_network(self, ckpt: str, load_D=False):
        """ Load the generator network from memory"""
        print(f"> Loading gan_wrappers from '{bcolors.OKBLUE}{ckpt}{bcolors.ENDC}'.")
        self.ckpt = ckpt
        with dnnlib.util.open_url(ckpt) as f:
            content = legacy.load_network_pkl(f)
            G = content['G_ema'].to(self.device)
            G = G.requires_grad_(False).eval()

            D = content['D'].to(self.device).eval()
            self.D = D.requires_grad_(False)
        self.G = G
        return G, content["training_set_kwargs"]

    def load_network_from_dict(self, content: dict):
        """ Load the generator network from a dictionary. """
        G = content['G_ema'].to(self.device)
        self.G = G.requires_grad_(False).eval()
        return G, content["training_set_kwargs"]

    def load_network_dict(self, ckpt: str):
        """ Load the generator network dict """
        print('Loading gan_wrappers from "%s"...' % ckpt)
        with src.stylegan3.dnnlib.util.open_url(ckpt) as f:
            ckpt = legacy.load_network_pkl(f)
        return ckpt

    def get_num_latents(self):
        return self.G.mapping.num_ws

    def get_w_dim(self):
        return self.G.mapping.w_dim

    def sample_latents(self,
                       batch_size: int = 32,
                       truncation_psi: float = 1.0,
                       c=None):
        seed = torch.randint(0, 2 ** 32, (1,))
        z = np.random.RandomState(seed).randn(batch_size, self.G.z_dim)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), c)[:, :1, :]  # [1, 1, C]; fake w_avg
        w = self.G.mapping.w_avg + truncation_psi * (w - self.G.mapping.w_avg)
        w = w.repeat([1, self.G.mapping.num_ws, 1])
        return w

    def generate(self, batch_size=32, truncation_psi=1.0, w=None, expand=False, **kwargs):
        """ Generates random images. """
        if w is None:
            w = self.sample_latents(batch_size=batch_size, truncation_psi=truncation_psi)

        if expand:
            synth_images = self.G.synthesis(w.unsqueeze(1).repeat([1, self.G.mapping.num_ws, 1]),
                                            noise_mode='const')
        else:
            synth_images = self.G.synthesis(w, noise_mode='const')
        return w, synth_images

    def parameters(self):
        """ Return all parameters of the generator """
        return self.G.synthesis.parameters()

    def named_parameters(self):
        return self.G.named_parameters()

    def activate_gradients(self):
        for param in self.parameters():
            param.requires_grad_(True)

    def save(self, **kwargs) -> dict:
        """ Serialize generator into dict. """
        snapshot_data = dict(G=None, D=None, G_ema=self.G, augment_pipe=None,
                             training_set_kwargs=dict())
        for key, value in snapshot_data.items():
            if isinstance(value, torch.nn.Module):
                value = copy.deepcopy(value).eval().requires_grad_(False)
                snapshot_data[key] = value.cpu()
            del value  # conserve memory

        save_dict = {
            ModelArgs.MODEL_KEY: snapshot_data,
            ModelArgs.MODEL_ARGS_KEY: self.model_args
        }
        return save_dict
