import os
from dataclasses import dataclass, field


@dataclass
class DetectArgs:
    CONFIG_KEY = "detect_args"
    """  Detect Image Arguments """

    watermark_key_ckpt: str = field(default=None, metadata={
        "help": "the checkpoint to the watermarking key."
    })

    image_folder: str = field(default=None, metadata={
        "help": "the folder containing the images to detect."
    })


