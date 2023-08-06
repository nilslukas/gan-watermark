import numpy as np
import torch

from src.model_converter.mappers.layer_wm_mapper import LayerWMMapper

from src.compatibility.stylegan_xl.torch_utils import misc, persistence
from src.compatibility.stylegan_xl.torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma


@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        padding=0,  # Padding with respect to the upsampled image.
        resample_filter=None,
        # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        demodulate=True,  # Apply weight demodulation?
        flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1, 2, 3],
                                                                            keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
                                        groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise.to(x.device))
    return x


@persistence.persistent_class
class SG2MessageSynthesisLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this layer.
                 weight,  # weight of the original layer
                 bias,  # bias of the original layer
                 affine,  # affine of the original layer
                 layer_mapper: LayerWMMapper,  # Mapper for the watermark
                 kernel_size=3,  # Convolution kernel size.
                 up=1,  # Integer upsampling factor.
                 use_noise=True,  # Enable noise input?
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 channels_last=False,  # Use channels_last format for the weights?
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.layer_mapper: LayerWMMapper = layer_mapper

        self.affine = affine
        self.weight = weight
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = bias

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution],
                                device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster

        # Watermark add-in (ugly, but we have to do it per-element)
        r = []
        modulation_dict: dict = self.layer_mapper(self.layer_mapper.get_msg(), styles)
        for i, (x_i, style_i, msg_i) in enumerate(zip(x, styles, self.layer_mapper.get_msg())):
            style_i = style_i + 0.01 * (modulation_dict['style'][i] if 'style' in modulation_dict else 0)
            modulated_bias = self.bias + 0.01 * (modulation_dict['bias'][i] if 'bias' in modulation_dict else 0)
            modulated_weight = self.weight * (
                        1 + 0.01 * (modulation_dict['weight'][i] if 'weight' in modulation_dict else 0))

            r += [modulated_conv2d(x=x_i.unsqueeze(0), weight=modulated_weight, styles=style_i.unsqueeze(0),
                                   noise=noise if noise_mode == 'const' else noise[i], up=self.up,
                                   padding=self.padding, resample_filter=self.resample_filter.to(x.device),
                                   flip_weight=flip_weight,
                                   fused_modconv=fused_modconv).squeeze()]
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            r[-1] = bias_act.bias_act(r[-1].unsqueeze(0), modulated_bias.to(r[-1].dtype), act=self.activation,
                                      gain=act_gain, clamp=act_clamp).squeeze()
        r = torch.stack(r, 0)
        return r

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])
