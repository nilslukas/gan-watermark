from typing import Callable

from src.arguments.model_args import ModelArgs


class GAN:

    def __init__(self, model_args: ModelArgs):
        self.model_args = model_args

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def get_num_latents(self):
        raise NotImplementedError

    def get_w_dim(self):
        raise NotImplementedError

    def activate_gradients(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def get_discriminator(self):
        raise NotImplementedError

    def save(self, **kwargs) -> dict:
        """ Saves the GAN. Returns a state dict """
        raise NotImplementedError

    def clear_post_processing(self):
        """ Clears all post-processing function to each sytnehsized image
        """
        raise NotImplementedError

    def add_post_processing(self, post_processing_fn: Callable):
        """ Adds a post-processing function to each sytnehsized image
        """
        raise NotImplementedError