import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.embed_args import EmbedArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.models.model_factory import ModelFactory
from src.trainer.ptw_trainer import PTWTrainer
from src.watermarking_key.wm_key import WatermarkingKey
from src.watermarking_key.wm_key_factory import WatermarkingKeyFactory


def parse_args():
    """ Embed a watermark using a pre-trained watermark decoder.
    """
    parser = transformers.HfArgumentParser((EmbedArgs,  # arguments for the embedding
                                            WatermarkingKeyArgs,  # arguments for the key
                                            ModelArgs,  # arguments for the generator
                                            EnvArgs,  # environment arguments
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def embed_from_pretrained(embed_args: EmbedArgs,
                          wm_key_args: WatermarkingKeyArgs,
                          model_args: ModelArgs,
                          env_args: EnvArgs,
                          config_args: ConfigArgs):
    """ Given any generator and any watermarking key, embed a watermark.
    """
    if config_args.exists():
        embed_args = config_args.get_embed_args()
        wm_key_args = config_args.get_watermarking_key_args()
        model_args = config_args.get_model_args()
        env_args = config_args.get_env_args()

    # load the pre-trained watermarking key
    wm_key: WatermarkingKey = WatermarkingKeyFactory.from_watermark_key_args(wm_key_args, env_args=env_args)
    wm_key.load(wm_key_args.key_ckpt)

    # load the generator.
    model, _ = ModelFactory.from_model_args(model_args)

    ptw = PTWTrainer(embed_args=embed_args, env_args=env_args)
    ptw.train(model, wm_key)     # will be an infinity loop

if __name__ == "__main__":
    embed_from_pretrained(*parse_args())
