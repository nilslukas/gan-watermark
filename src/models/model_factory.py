from typing import Tuple

import torch

from src.arguments.env_args import EnvArgs
from src.models.stylegan import StyleGAN
from src.utils.highlited_print import print_highlighted

from src.arguments.model_args import ModelArgs, ModelTypes
from src.models.wm_decoder import WatermarkDecoder


class ModelFactory:
    @staticmethod
    def load_decoder(bitlen: int, decoder_arch: str) -> WatermarkDecoder:
        """ Loads a model from model_args. If the model checkpoint field is set, this function loads
        the model and its arguments from that location """
        return WatermarkDecoder(bitlen=bitlen, decoder_arch=decoder_arch)

    @staticmethod
    def from_model_args(model_args: ModelArgs) -> Tuple[StyleGAN, dict]:
        """ Loads a model from model_args. If the model checkpoint field is set, this function loads
        the model and its arguments from that location """
        if model_args.model_type == ModelTypes.STYLEGAN.value:
            gan = StyleGAN(model_args)
        else:
            raise ValueError(model_args.model_type)

        kwargs = {}
        if model_args.model_ckpt is not None:
            print_highlighted(f"> {model_args}")
            if model_args.model_ckpt.endswith(".pt"):
                content = torch.load(model_args.model_ckpt)[ModelArgs.MODEL_KEY]
                _, kwargs = gan.load_network_from_dict(content)
            else:
                _, kwargs = gan.load_network(model_args.model_ckpt, load_D=model_args.load_discriminator)
        return gan, kwargs
