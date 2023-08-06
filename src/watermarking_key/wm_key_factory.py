from src.arguments.env_args import EnvArgs
from src.arguments.wm_key_args import WatermarkingKeyArgs, WatermarkType
from src.watermarking_key.ptw_wm_key import PTWWatermarkingKey
from src.watermarking_key.wm_key import WatermarkingKey


class WatermarkingKeyFactory:
    """ Generate watermarking keys from configuration files.
    """
    @staticmethod
    def from_watermark_key_args(watermark_args: WatermarkingKeyArgs, env_args: EnvArgs = None) -> WatermarkingKey:
        print(f"> Loading watermarking key args: '{watermark_args.watermark_type}'")
        if watermark_args.watermark_type == WatermarkType.PTW.value:
            """ The watermark by Lukas et al. 
             @Desc: mark generator weights and inputs. 
            """
            watermark = PTWWatermarkingKey(watermark_args, env_args=env_args)
        else:
            raise ValueError(watermark_args.watermark_type)

        return watermark
