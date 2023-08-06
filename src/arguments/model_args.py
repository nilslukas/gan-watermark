from dataclasses import dataclass, field
from enum import Enum

class ModelTypes(Enum):
    STYLEGAN = "nvlabs"

@dataclass
class ModelArgs:
    CONFIG_KEY = "model_args"
    """ This class contains all parameters for the generator. """

    # The following fields denote the key names in a checkpoint file.
    MODEL_KEY = "gan"
    MODEL_ARGS_KEY = "gan_args"

    model_type: str = field(default="nvlabs", metadata={
        "help": "model type",
        "choices": ["nvlabs"]
    })

    model_arch: str = field(default=None, metadata={
        "help": "model architecture (stylegan2, stylegan3, stylegan-xl). Optional.",
    })

    model_ckpt: str = field(default=None, metadata={
        "help": "path to the weights of the pre-trained checkpoint (if applicable)"
    })

    load_discriminator: bool = field(default=False, metadata={
        "help": "whether to load the discriminator network",
    })





