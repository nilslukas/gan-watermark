import os
from dataclasses import dataclass, field


@dataclass
class GenerateImageArgs:
    CONFIG_KEY = "gen_img_args"
    """  Generate Image Arguments """

    generator_ckpt: str = field(default=None, metadata={
        "help": "the checkpoint to the generator."
    })

    outdir: str = field(default="../generated_images/your_folder", metadata={
        "help": "the directory to save the generated images."
    })

    num_images: int = field(default=1_000, metadata={
        "help": "number of images to generate"
    })

    truncation_psi: float = field(default=0.7, metadata={
        "help": "truncation psi to generate images."
    })


