import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.models.model_factory import ModelFactory
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory
from src.watermarking_key.wm_key import WatermarkingKey


def parse_args():
    """ Train a watermarking key (i.e., a decoder network)
    """
    parser = transformers.HfArgumentParser((WatermarkingKeyArgs,
                                            ModelArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def keygen(wm_key_args: WatermarkingKeyArgs,
           model_args: ModelArgs,
           env_args: EnvArgs,
           config_args: ConfigArgs):
    """ Trains a watermarking decoder key.
    """
    if config_args.exists():  # a configuration file was provided. yaml files always overwrite other settings!
        wm_key_args = config_args.get_watermarking_key_args()  # params to instantiate the watermarking key.
        model_args = config_args.get_model_args()  # params to instantiate the generator.
        env_args = config_args.get_env_args()

    model, _ = ModelFactory.from_model_args(model_args)
    wm_key: WatermarkingKey = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args)
    wm_key.learn(model)


if __name__ == "__main__":
    keygen(*parse_args())
