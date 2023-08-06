import numpy as np
import torch
from src.compatibility.stylegan_xl.torch_utils import misc
from src.compatibility.stylegan_xl.torch_utils.ops import filtered_lrelu, conv2d_gradfix

from src.model_converter.mappers.layer_wm_mapper import LayerWMMapper


@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
        s,  # Style tensor: [batch_size, in_channels]
        demodulate=True,  # Apply weight demodulation?
        padding=0,  # Padding: int or [padH, padW]
        input_gain=None,
        # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class SG3MessageSynthesisLayer(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 is_torgb,  # Is this the final ToRGB layer?
                 is_critically_sampled,  # Does this layer use critical sampling?
                 use_fp16,  # Does this layer use FP16?

                 layer_mapper: LayerWMMapper,  # Mapper for the watermark
                 affine,
                 weight,
                 bias,

                 # Input & output specifications.
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 in_size,  # Input spatial size: int or [width, height].
                 out_size,  # Output spatial size: int or [width, height].
                 in_sampling_rate,  # Input sampling rate (s).
                 out_sampling_rate,  # Output sampling rate (s).
                 in_cutoff,  # Input cutoff frequency (f_c).
                 out_cutoff,  # Output cutoff frequency (f_c).
                 in_half_width,  # Input transition band half-width (f_h).
                 out_half_width,  # Output Transition band half-width (f_h).

                 up_taps,
                 down_taps,
                 magnitude_ema,
                 up_filter,
                 down_filter,
                 padding,
                 up_factor,
                 down_factor,
                 tmp_sampling_rate,
                 down_radial,

                 # Hyperparameters.
                 conv_kernel=3,  # Convolution kernel size. Ignored for final the ToRGB layer.
                 filter_size=6,  # Low-pass filter size relative to the lower resolution when up/downsampling.
                 lrelu_upsampling=2,  # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
                 use_radial_filters=False,
                 # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
                 conv_clamp=256,  # Clamp the output to [-X, +X], None = disable clamping.
                 magnitude_ema_beta=0.999,  # Decay rate for the moving average of input magnitudes.
                 ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = tmp_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        self.layer_mapper: LayerWMMapper = layer_mapper

        # Setup parameters and buffers.
        self.affine = affine  # FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = weight  # torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = bias  # torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', magnitude_ema)

        # Design upsampling filter.
        self.up_factor = up_factor
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = up_taps
        self.register_buffer('up_filter', up_filter)

        # Design downsampling filter.
        self.down_factor = down_factor
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = down_taps
        self.down_radial = down_radial
        self.register_buffer('down_filter', down_filter)

        self.padding = padding

    def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
        assert noise_mode in ['random', 'const', 'none']  # unused
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        # Watermark add-in (ugly, but we have to do it per element)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        new_x = []
        modulation_dict: dict = self.layer_mapper(self.layer_mapper.get_msg(), styles)
        for i, (x_i, style_i, msg_i) in enumerate(zip(x.to(dtype), styles, self.layer_mapper.get_msg())):

            style_i = style_i + 0.01 * (modulation_dict['style'][i] if 'style' in modulation_dict else 0)
            modulated_bias = self.bias + 0.01 * (modulation_dict['bias'][i] if 'bias' in modulation_dict else 0)
            modulated_weight = self.weight * (
                    1 + 0.01 * (modulation_dict['weight'][i] if 'weight' in modulation_dict else 0))

            x_i = modulated_conv2d(x=x_i.unsqueeze(0).to(dtype), w=modulated_weight, s=style_i.unsqueeze(0),
                                   padding=self.conv_kernel - 1, demodulate=(not self.is_torgb),
                                   input_gain=input_gain).to(x.dtype)

            # Execute bias, filtered leaky ReLU, and clamping.
            gain = 1 if self.is_torgb else np.sqrt(2)
            slope = 1 if self.is_torgb else 0.2
            x_i = filtered_lrelu.filtered_lrelu(x=x_i, fu=self.up_filter, fd=self.down_filter,
                                                b=modulated_bias.to(x.dtype),
                                                up=self.up_factor, down=self.down_factor, padding=self.padding,
                                                gain=gain, slope=slope, clamp=self.conv_clamp)
            new_x += [x_i.squeeze()]
        # Ensure correct shape and dtype.
        new_x = torch.stack(new_x, 0).to(dtype)
        misc.assert_shape(new_x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert new_x.dtype == dtype
        return new_x
