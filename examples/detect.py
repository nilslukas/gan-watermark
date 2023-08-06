import os
import pprint

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.arguments.config_args import ConfigArgs
from src.arguments.detect_args import DetectArgs
from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.utils.highlited_print import bcolors
from src.watermarking_key.wm_key import WatermarkingKey
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def parse_args():
    parser = transformers.HfArgumentParser((DetectArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if
                            os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith('.png')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)

        # Load the image using PIL (you can replace this with your preferred library)
        image = Image.open(image_path).convert('RGB')

        # Apply transformations to convert to tensor and normalize to [-1, 1]
        image = self.transform(image)

        return image, image_file


def detect_watermark(detect_args: DetectArgs,
                    env_args: EnvArgs,
                    config_args: ConfigArgs):
    """ Load a generator and show some images. """
    if config_args.exists():
        detect_args = config_args.get_detect_args()
        env_args = config_args.get_env_args()

    # load the watermarking key.
    ckpt_fn = os.path.abspath(detect_args.watermark_key_ckpt)
    print(f"> Loading watermarking key from '{bcolors.OKBLUE}{ckpt_fn}{bcolors.ENDC}'.")
    wm_key_args: WatermarkingKeyArgs = torch.load(ckpt_fn)[WatermarkingKeyArgs.WM_KEY_ARGS_KEY]
    wm_key_args.key_ckpt = ckpt_fn
    wm_key: WatermarkingKey = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args)
    wm_key.to(env_args.device)
    wm_key.eval()

    # load images from file path
    dataset = ImageDataset(detect_args.image_folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=env_args.batch_size, shuffle=False, num_workers=4)

    y_true = WatermarkingKey.str_to_bits(wm_key.wm_key_args.message).unsqueeze(0)[:, :wm_key_args.bitlen]
    y_true = y_true.repeat([env_args.batch_size, 1]).to(env_args.device)

    result_dict = {}
    for x, fps in tqdm(dataloader, total=len(dataloader), desc="Detecting Watermark"):
        x = x.to(env_args.device)
        y_pred = wm_key.extract(x, sigmoid=True)

        for j, (fp, pred) in enumerate(zip(fps, y_pred)):
            result_dict[fp] = {
                "message": WatermarkingKey.bits_to_str(torch.round(pred).int().cpu()),
                "accuracy": round(pred.round().eq(y_true[j].float()).float().mean().item(), 2)
            }
    pprint.pprint(result_dict)
    mean_accuracy = np.mean([result_dict[fp]["accuracy"] for fp in result_dict])
    print(f"> Mean Bit Accuracy: {mean_accuracy:.2f}")

if __name__ == "__main__":
    detect_watermark(*parse_args())
