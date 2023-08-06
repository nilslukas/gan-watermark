import os.path
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

from src.utils.highlited_print import bcolors


class WatermarkType(Enum):
    PTW = "ptw"  # ours, SOTA

@dataclass
class WatermarkingKeyArgs:
    CONFIG_KEY = "watermarking_key_args"
    """ This class contains all arguments for the watermarking key. """

    WM_KEY_ARGS_KEY = "watermarking_args_key"    # field to save in the dictionary.
    ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%"  # alphabet to convert messages from bits to string

    def __post_init__(self):
        if self.message is None:
            self.message = ''.join(np.random.choice(list(WatermarkingKeyArgs.ALPHABET), size=100))

        if self.key_ckpt is not None and os.path.exists(self.key_ckpt):  # attempt to load your own watermarking params
            data = torch.load(self.key_ckpt)
            key = self.key_ckpt
            print(f"> Restoring watermark arguments from '{bcolors.OKGREEN}{os.path.abspath(self.key_ckpt)}{bcolors.ENDC}'")
            self.__dict__ = vars(data[self.WM_KEY_ARGS_KEY])
            self.key_ckpt = key

    bitlen: int = field(default=50, metadata={
        "help": "bit length of the watermark."
    })

    message: str = field(default=None, metadata={
        "help": "the embedded_message to embed. Will be converted into a (truncated) binary sequence."
                " Only used during the embedding stage."
    })

    key_ckpt: str = field(default=None, metadata={
        "help": "path to a pre-trained checkpoint. If given, this function will attempt to load the watermarking"
                "args from the checkpoint using the {WM_ARGS_KEY} key entry."
    })

    keygen_lambda_lpips: float = field(default=0.01, metadata={
        "help": "lambda for the lpips loss"
    })
    
    keygen_lambda_id: float = field(default=0.01, metadata={
        "help": "lambda for the facial id loss"
    })

    lr_mapper: float = field(default=0.001, metadata={
        "help": "learning rate for the mapper"
    })

    lr_decoder: float = field(default=0.0001, metadata={
        "help": "learning rate for the detector"
    })

    watermark_type: str = field(default=WatermarkType.PTW.value, metadata={
        "help": "type of the watermarking method to use. ",
        "choices": [t.value for t in WatermarkType]
    })

    decoder_arch: str = field(default="resnet18", metadata={
        "help": "model architecture of the decoder",
        "choices": ["resnet18", "resnet50", "resnet101"]
    })

    first_marked_layer: int = field(default=0, metadata={
        "help": "the first watermarked layer (0=all layers will be watermarked)"
    })

    ir_se50_weights: str = field(default=None, metadata={
        "help": "path to a facial recognition model."
                "download from here: https://github.com/nvlong21/Face_Recognize"
    })

    truncation_psi: float = field(default=.7, metadata={
        "help": "global truncation during embedding. larger values lead to a longer embedding time, but "
                "also more robust watermark. "
    })

    weight_mapper: bool = field(default=False, metadata={
        "help": "whether to add a style mapper to the input. "
    })

    bias_mapper: bool = field(default=False, metadata={
        "help": "whether to add a bias mapper to the deconvolution layers"
    })

    style_mapper: bool = field(default=False, metadata={
        "help": "whether to add a style mapper to the input. "
    })

    add_preprocessing: bool = field(default=False, metadata={
        "help": "add pre-processing to training"
    })



