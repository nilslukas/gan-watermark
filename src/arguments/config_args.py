from dataclasses import dataclass, field

import yaml

from src.arguments.detect_args import DetectArgs
from src.arguments.embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs
from src.arguments.generate_image_args import GenerateImageArgs
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.utils.highlited_print import print_warning


@dataclass
class ConfigArgs:

    config_path: str = field(default=None, metadata={
        "help": "path to the yaml configuration file (*.yml)"
    })

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify these keys in the *.yml file
        WatermarkingKeyArgs.CONFIG_KEY: WatermarkingKeyArgs(),
        EnvArgs.CONFIG_KEY: EnvArgs(),
        ModelArgs.CONFIG_KEY: ModelArgs(),
        EmbedArgs.CONFIG_KEY: EmbedArgs(),
        GenerateImageArgs.CONFIG_KEY: GenerateImageArgs(),
        DetectArgs.CONFIG_KEY: DetectArgs()
    }

    def get_embed_args(self) -> EmbedArgs:
        return self.args_to_config[EmbedArgs.CONFIG_KEY]

    def get_watermarking_key_args(self) -> WatermarkingKeyArgs:
        content: WatermarkingKeyArgs = self.args_to_config[WatermarkingKeyArgs.CONFIG_KEY]
        return content

    def get_detect_args(self) -> DetectArgs:
        return self.args_to_config[DetectArgs.CONFIG_KEY]

    def get_generate_image_args(self) -> GenerateImageArgs:
        return self.args_to_config[GenerateImageArgs.CONFIG_KEY]

    def get_env_args(self) -> EnvArgs:
        return self.args_to_config[EnvArgs.CONFIG_KEY]

    def get_model_args(self) -> ModelArgs:
        return self.args_to_config[ModelArgs.CONFIG_KEY]

    def __post_init__(self):
        if self.config_path is None:
            print("> No config file specified. Using default values.")
            return

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.keys = list(data.keys())

        # load arguments
        for i in range(2):  # naive solution: Fill in keys, call post-init, fill in keys again
            keys_not_found = []
            for entry, values in data.items():
                for key, value in values.items():
                    if key not in self.args_to_config[entry].__dict__.keys():
                        keys_not_found += [(entry, key)]
                    self.args_to_config[entry].__dict__[key] = value
            if i == 0:
                if len(keys_not_found) > 0:
                    print_warning(f"Could not find these keys: {keys_not_found}. Make sure they exist.")

            if i == 0:
                for key, value in self.args_to_config.items():
                    value.__setattr__("config_path", self.config_path)
                    if hasattr(value, "__post_init__"):
                        value.__post_init__()







