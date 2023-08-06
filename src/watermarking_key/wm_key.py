import os.path
from typing import Union

import numpy as np
import torch
from torch import nn

from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.utils.highlited_print import bcolors


class WatermarkingKey(nn.Module):

    def __init__(self, wm_key_args: WatermarkingKeyArgs, env_args: EnvArgs = None):
        """ The base class for a watermarking key. """
        super().__init__()
        self.wm_key_args = wm_key_args
        self.env_args = EnvArgs() if env_args is None else env_args  # assume default env arguments if none are given.

    @staticmethod
    def bits_to_str(bits: torch.Tensor) -> str:
        """ Convert bits to human-readable string """
        if isinstance(bits, str):
            return bits
        msg = ""
        for i in range(int(np.ceil(len(bits) / 5))):
            index = [str(x.item()) for x in bits[i * 5:(i + 1) * 5]]
            index = ''.join(map(str, index))
            index = int(index, 2)
            msg += WatermarkingKeyArgs.ALPHABET[index]
        return msg

    @staticmethod
    def str_to_bits(msg: str) -> torch.Tensor:
        """ Convert human-readable string to bits """
        if isinstance(msg, torch.Tensor):
            return msg
        msg = msg.upper()    # only uppercase
        print(f"> Converting message '{bcolors.OKGREEN}{msg}{bcolors.ENDC}' to bits.")
        bits = torch.zeros(size=(len(msg) * 5,))
        for i, letter in enumerate(msg):
            try:
                pos = WatermarkingKeyArgs.ALPHABET.index(letter)
            except ValueError:
                raise ValueError(f"Letter '{letter}' is not in the alphabet ('{bcolors.OKGREEN}{WatermarkingKeyArgs.ALPHABET}{bcolors.ENDC}').")
            bitstr = '{0:05b}'.format(pos)
            for j, x in enumerate(bitstr):
                bits[i * 5 + j] = int(x)
        return bits

    def save(self, ckpt_fn: str = None) -> dict:
        """ Saves a key to a single '*.pt' file. If no ckpt_fn is given, only returns the save dict."""
        save_dict = {
            WatermarkingKeyArgs.WM_KEY_ARGS_KEY: self.wm_key_args
        }
        if ckpt_fn is not None:
            print(f"> Saving Watermarking Decoder checkpoint to '{bcolors.OKGREEN}{os.path.abspath(ckpt_fn)}{bcolors.ENDC}'")
            torch.save(save_dict, ckpt_fn)
        return save_dict

    def load(self, ckpt=None):
        """ Loads a key from a '*.pt' file. """
        raise NotImplementedError

    def gen_msg(self, n: int) -> torch.Tensor:
        """ Generate n random binary messages.  """
        return torch.randint(0, 2, size=(n, self.wm_key_args.bitlen))

    def extract(self, x: torch.Tensor, sigmoid=True, **kwargs):
        """
        Extracts a embedded_message from one or more images.
        Note: Sigmoid can be turned off if used to compute the loss.
        """
        raise NotImplementedError

    def validate(self, x: torch.Tensor, msg: Union[str, torch.Tensor]):
        """
        Extracts a embedded_message from one or more images and computes the mean bit accuracy.
        """

        if isinstance(msg, str):
            msg = WatermarkingKey.str_to_bits(msg).unsqueeze(0).repeat([x.shape[0], 1])

        msg_pred = self.extract(x, sigmoid=True)
        msg_pred[msg_pred >= 0.5] = 1
        msg_pred[msg_pred < 0.5] = 0
        bitwise_acc = (msg_pred == msg[:, :self.wm_key_args.bitlen].to(x.device)).float().mean(dim=1).mean().item()
        return bitwise_acc

    def loss(self, x: torch.Tensor, msg: torch.Tensor):
        """
        Given images and a embedded_message, compute the loss.
        """
        extracted_msg = self.extract(x, sigmoid=False)
        bitlength = min(self.wm_key_args.bitlen, msg.shape[1])
        return nn.BCEWithLogitsLoss()(extracted_msg[:, :bitlength], msg[:, :bitlength].to(extracted_msg.device))


