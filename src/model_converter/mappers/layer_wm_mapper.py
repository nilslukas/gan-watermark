import torch
import math

from src.arguments.env_args import EnvArgs
from src.model_converter.mappers.wm_mapper import WMMapper
from src.utils.equal_linear import EqualLinear
from src.utils.highlited_print import print_dict_highlighted


class LayerWMMapper(WMMapper):
    def __init__(self, bitlength: int, backbone=None, style_dim=None, hidden_layers: int=1, env_args: EnvArgs = None):
        super().__init__(bitlength)
        self.style_dim = 512 if style_dim is None else style_dim
        self.hidden_layers = hidden_layers
        self.backbone = backbone
        self.heads = []

        self.bias_shape = None
        self.bias_head = None
        self.style_shape = None
        self.style_head = None
        self.noise_shape = None
        self.noise_head = None
        self.weight_shape = None
        self.weight_head = None
        self.env_args = env_args if env_args is not None else EnvArgs()

    def summary(self):
        data = {"LayerWMMapper": {"bit_length": self.bitlength, "style_dim": self.style_dim}}
        if self.bias_head:
            data['bias'] = self.bias_shape
        if self.style_head:
            data['style'] = self.style_shape
        if self.weight_head:
            data['weight'] = self.weight_shape
        print_dict_highlighted(data)

    def add_bias_mapper(self, shape):
        self.bias_shape = shape

    def add_style_mapper(self, shape):
        self.style_shape = shape

    def add_noise_mapper(self, shape):
        self.noise_shape = shape

    def add_weight_mapper(self, shape):
        self.weight_shape = shape

    def build(self):
        hidden_dim = self.style_dim + self.bitlength
        if self.backbone is None:
            self.backbone = torch.nn.Sequential(*[
                EqualLinear(hidden_dim, hidden_dim) if self.hidden_layers > 0 else torch.nn.Identity(),
                EqualLinear(hidden_dim, self.style_dim),
            ]).to(self.env_args.device)  # project the latent code + embedded_message

        # one projection head per output (n, embedded_message + style) -> (n, output)
        if self.bias_shape:
            self.bias_head = torch.nn.Sequential(*[
                torch.nn.Linear(self.bitlength + self.style_dim, math.prod(self.bias_shape))
            ]).to(self.env_args.device)
        if self.weight_shape:
            self.weight_head = torch.nn.Sequential(*[
                torch.nn.Linear(self.bitlength + self.style_dim, math.prod(self.weight_shape))
            ]).to(self.env_args.device)
        if self.style_shape:
            self.style_head = torch.nn.Sequential(*[
                torch.nn.Linear(self.bitlength + self.style_dim, math.prod(self.style_shape))
            ]).to(self.env_args.device)

    def forward(self, msg: torch.Tensor, style: torch.Tensor) -> dict:
        """ Returns the head outputs for a embedded_message and a style
        """
        return_dict = {}
        feats = self.backbone(torch.cat([msg.float(), style], dim=-1))

        if self.bias_head:
            return_dict["bias"] = self.bias_head(torch.cat([msg, feats], dim=-1)).view([-1, *self.bias_shape])
        if self.weight_head:
            return_dict["weight"] = self.weight_head(torch.cat([msg, feats], dim=-1)).view([-1, *self.weight_shape])
        if self.style_head:
            return_dict["style"] = self.style_head(torch.cat([msg, feats], dim=-1)).view([-1, *self.style_shape])
        return return_dict


class WMMapperGroup:
    def __init__(self, env_args: EnvArgs = None):
        self.mappers = []
        self.env_args = env_args if env_args is not None else EnvArgs()

    def add_mapper(self, mapper: WMMapper):
        self.mappers += [mapper]

    def build(self):
        for mapper in self.mappers:
            mapper.build()
        return self

    def set_msg(self, msg: torch.Tensor):
        for mapper in self.mappers:
            mapper.set_msg(msg)

    def size(self):
        return len(self.mappers)

    def __iter__(self):
        for mapper in self.mappers:
            yield mapper