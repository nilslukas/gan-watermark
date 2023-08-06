from copy import deepcopy
from typing import Tuple

from tqdm import tqdm
from src.utils.highlited_print import bcolors

from src.arguments.wm_key_args import WatermarkingKeyArgs
from src.model_converter.mappers.layer_wm_mapper import WMMapperGroup, LayerWMMapper
from src.model_converter.wm_layers.sg2_watermarked import SG2MessageSynthesisLayer
from src.model_converter.wm_layers.sg3_watermarked import SG3MessageSynthesisLayer
from src.models.gan import GAN


class ModelConverter:
    """ This class converts a vanilla StyleGAN2, StyleGAN-XL and StyleGAN3 into
    a version that can be watermarked. """

    def __init__(self, wm_key_args: WatermarkingKeyArgs):
        self.wm_key_args = wm_key_args

    def detect_model_arch(self, model: GAN) -> str:
        if model.model_args.model_arch is not None:
            print(f"> Model architecture is {bcolors.OKGREEN}{model.model_args.model_arch}{bcolors.ENDC}.")
            return model.model_args.model_arch

        names = [name for name, layer in model.G.synthesis.named_children()]
        print(f"> No model arch provided. Resorting to guessing. Detected layer names: {names}")

        print(model.G.synthesis)
        if 'b4' in names and 'b8' in names:
            print(f"> Best guess is {bcolors.OKGREEN}NVLabs - StyleGAN2{bcolors.ENDC}.")
            return "stylegan2"
        elif 'L0_36_512' in names or 'L1_36_1024' in names:
            print(f"> Best guess is {bcolors.OKGREEN}NVLabs - StyleGAN3{bcolors.ENDC}.")
            return "stylegan3"
        elif 'L1_36_1024' in names:
            print(f"> Best guess is {bcolors.OKGREEN}StyleGAN-XL{bcolors.ENDC}.")
            return "stylegan-xl"

    def _build_sg2(self, model: GAN) -> Tuple[GAN, WMMapperGroup]:
        """ Watermark a StyleGAN2 model. """
        mapper_group = WMMapperGroup()

        print(f"> Converting {bcolors.OKGREEN}StyleGAN2{bcolors.ENDC} to watermarked.")
        for layer_ctr, (name, child) in enumerate(tqdm(model.G.synthesis.named_children(), desc="Model Conversion",
                                                       total=len(list(model.G.synthesis.children())))):
            if layer_ctr < self.wm_key_args.first_marked_layer:
                continue     # skip until first_marked layer is reached.
            for gc_name, gc in child.named_children():
                if not gc_name in ["conv0", "conv1"]:  # only convert those two layers.
                    continue

                mapper = LayerWMMapper(bitlength=self.wm_key_args.bitlen, style_dim=gc.affine.out_features)  # instantiate the mapper.

                mapper.add_bias_mapper(gc.bias.shape) if self.wm_key_args.bias_mapper else None
                mapper.add_style_mapper((gc.weight.shape[1],)) if self.wm_key_args.style_mapper else None
                mapper.add_weight_mapper((*gc.weight.shape[:2], 1, 1)) if self.wm_key_args.weight_mapper else None

                mapper.build()  # materialize the mapper
                #mapper.summary()

                # now replace the whole layer
                alternate_layer = SG2MessageSynthesisLayer(in_channels=gc.weight.shape[1], out_channels=gc.weight.shape[0],
                                                           w_dim=gc.w_dim if hasattr(gc, "w_dim") else gc._init_kwargs['w_dim'],
                                                           up=gc.up, use_noise=gc.use_noise, activation=gc.activation,
                                                           kernel_size=gc.weight.shape[-1], resolution=gc.resolution,
                                                           weight=deepcopy(gc.weight), affine=deepcopy(gc.affine),
                                                           bias=deepcopy(gc.bias), resample_filter=deepcopy(gc.resample_filter),
                                                           layer_mapper=mapper)

                child.__setattr__(gc_name, alternate_layer)
                mapper_group.add_mapper(mapper)
                del gc
        print(f"> Watermarked: {mapper_group.size()} layers!")
        return model, mapper_group.build()

    def _build_sg3(self, model: GAN) -> Tuple[GAN, WMMapperGroup]:
        """ Watermark a StyleGAN2 model. """
        mapper_group = WMMapperGroup()

        print(f"> Converting {bcolors.OKGREEN}StyleGAN3{bcolors.ENDC} to watermarked.")
        for layer_ctr, (name, child) in enumerate(tqdm(model.G.synthesis.named_children(), desc="Model Conversion",
                                                       total=len(list(model.G.synthesis.children())))):
            if layer_ctr < self.wm_key_args.first_marked_layer:
                continue     # skip until first_marked layer is reached.

            if name in ["input"]:  # skip these layers
                continue

            mapper = LayerWMMapper(bitlength=self.wm_key_args.bitlen, style_dim=child.affine.out_features)  # instantiate the mapper.

            mapper.add_bias_mapper((*child.bias.shape,)) if self.wm_key_args.bias_mapper else None
            mapper.add_style_mapper((child.affine.out_features,)) if self.wm_key_args.style_mapper else None
            mapper.add_weight_mapper((*child.weight.shape[:2], 1, 1)) if self.wm_key_args.weight_mapper else None

            mapper.build()

            alternate_layer = SG3MessageSynthesisLayer(w_dim=child.w_dim,
                                                       is_torgb=child.is_torgb,
                                                       is_critically_sampled=child.is_critically_sampled,
                                                       use_fp16=child.use_fp16,
                                                       in_channels=child.in_channels,
                                                       out_channels=child.out_channels,
                                                       in_size=child.in_size,
                                                       out_size=child.out_size,
                                                       in_sampling_rate=child.in_sampling_rate,
                                                       out_sampling_rate=child.out_sampling_rate,
                                                       in_cutoff=child.in_cutoff,
                                                       out_cutoff=child.out_cutoff,
                                                       up_factor=child.up_factor,
                                                       down_factor=child.down_factor,
                                                       tmp_sampling_rate=child.tmp_sampling_rate,
                                                       in_half_width=child.in_half_width,
                                                       out_half_width=child.out_half_width,
                                                       conv_kernel=child.conv_kernel,
                                                       up_taps=child.up_taps,
                                                       down_taps=child.down_taps,
                                                       padding=child.padding,
                                                       magnitude_ema=child.magnitude_ema,
                                                       up_filter=child.up_filter,
                                                       down_filter=child.down_filter,
                                                       affine=child.affine,
                                                       weight=child.weight,
                                                       bias=child.bias,
                                                       magnitude_ema_beta=child.magnitude_ema_beta,
                                                       down_radial=child.down_radial,
                                                       layer_mapper=mapper)

            model.G.synthesis.__setattr__(name, alternate_layer)
            mapper_group.add_mapper(mapper)
            del child
        print(f"> Watermarked: {mapper_group.size()} layers!")
        return model, mapper_group.build()

    def convert(self, model: GAN) -> Tuple[GAN, WMMapperGroup]:
        """ Given a GAN, swap out layers according to the watermarking key argument"""
        arch = self.detect_model_arch(model)
        if arch == "stylegan2":
            return self._build_sg2(model)
        elif arch == "stylegan3":
            return self._build_sg3(model)
        elif arch == "stylegan-xl":
            raise NotImplementedError  # support will eventually follow.
        else:
            raise ValueError("Could not detect model type.")