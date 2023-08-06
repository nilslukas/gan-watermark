import os
from dataclasses import dataclass, field


@dataclass
class EnvArgs:
    CONFIG_KEY = "env_args"
    """ This class contains all arguments for the environment where to load samples. """

    logging_tool: str = field(default='wandb', metadata={
        "help": "tool to log experimental data with. Currently, only wandb is supported. ",
        "choices": [None, "wandb"]
    })

    log_every: int = field(default=100, metadata={
        "help": "log interval for training"
    })

    save_every: int = field(default=249, metadata={
        "help": "save interval for training"
    })

    device: str = field(default="cuda", metadata={
        "help": "device to run experiments on"
    })

    batch_size: int = field(default=16, metadata={
        "help": "default batch size for training"
    })

    eval_batch_size: int = field(default=128, metadata={
        "help": "default batch size for inference"
    })

    verbose: bool = field(default=True, metadata={
        "help": "whether to print out to the cmd line"
    })

    gradient_accumulation_steps: int = field(default=1, metadata={
        "help": "number of steps to accumulate gradients"
    })
