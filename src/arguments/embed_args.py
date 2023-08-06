from dataclasses import dataclass, field


@dataclass
class EmbedArgs:
    CONFIG_KEY = "embed_args"

    ckpt: str = field(default=None, metadata={
        "help": "the ckpt to persist or load the watermarked model"
    })

    ptw_lr: float = field(default=1e-3, metadata={
        "help": "learning rate for ptw"
    })

    lambda_lpips: float = field(default=1, metadata={
        "help": "lambda for lpips loss"
    })

    lambda_id: float = field(default=0.1, metadata={
        "help": "lambda for facial id loss"
    })





