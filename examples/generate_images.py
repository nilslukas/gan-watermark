import os

import numpy as np
import torch
import transformers
from torchvision.utils import save_image
from tqdm import tqdm

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.generate_image_args import GenerateImageArgs
from src.arguments.model_args import ModelArgs
from src.models.model_factory import ModelFactory
from src.utils.highlited_print import bcolors


def parse_args():
    parser = transformers.HfArgumentParser((GenerateImageArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def generate_images(generate_image_args: GenerateImageArgs,
                    env_args: EnvArgs,
                    config_args: ConfigArgs):
    """ Load a generator and save generated images to disk. """
    if config_args.exists():
        generate_image_args = config_args.get_generate_image_args()
        env_args = config_args.get_env_args()

    # load the model args
    ckpt = generate_image_args.generator_ckpt
    print(f"> Loading generator from '{bcolors.OKBLUE}{ckpt}{bcolors.ENDC}'.")
    model_args = torch.load(ckpt)[ModelArgs.MODEL_ARGS_KEY]
    model_args.model_ckpt = ckpt

    generator, kwargs = ModelFactory.from_model_args(model_args)

    # generate images
    num_images = generate_image_args.num_images
    outdir = os.path.abspath(generate_image_args.outdir)
    print(f"> Generating {num_images} images to '{bcolors.OKGREEN}{outdir}{bcolors.ENDC}'.")
    os.makedirs(outdir, exist_ok=True)
    ctr = 0
    for i in tqdm(range(int(np.ceil(num_images / env_args.batch_size))), desc="Write Images"):
        _, x = generator.generate(env_args.batch_size, truncation_psi=generate_image_args.truncation_psi)

        for j, x_i in enumerate(x):
            filename = os.path.join(outdir, f"image_{ctr}.png")
            save_image((x_i.cpu() + 1) / 2, filename)  # Save the image as PNG
            ctr += 1

if __name__ == "__main__":
    generate_images(*parse_args())
