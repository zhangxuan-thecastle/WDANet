import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from .blocks_2d import *
from .deform_ops import DeformConv2d
from .nat_2d import NeighborhoodAttention2D


class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


# MAB implementation
class MAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        '''
            channels: The number of filter at the output (usually the same with the number of filter from the input)
            kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
            dilation: The dilation dimension 'd' from the paper
            reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        '''
        super(MAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2*dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=2,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(AttentionBlock, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))

        c2wh = dict([(128, 56), (256, 28), (512, 14), (1024, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels], frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        feature_aggregation /= self.scale_branches
        feature_aggregation += x

        return feature_aggregation
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def wavelet_transform_init(filters):
    class WaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = inverse_wavelet_transform(grad_output, filters)
            return grad, None

    return WaveletTransform().apply

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MSC(nn.Module):
    """
    Multi-scale convolution block (MSC)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


#   Multi-scale convolution(MSC)
def MSCLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    msc = MSC(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(msc)
    if n > 1:
        for i in range(1, n):
            msc = MSC(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(msc)
    conv = nn.Sequential(*convs)
    return conv

def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wavelet_transform(grad_output, filters)
            return grad, None

    return InverseWaveletTransform().apply


class WDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',groups=1
                 ):
        super(WDConv2d, self).__init__()

        assert in_channels == out_channels #原本
        self.groups = groups
        self.in_channels = in_channels
        self.wt_levels = wt_levels #小波变换的层数
        self.stride = stride
        self.dilation = 1
        #小波变换（Wavelet Transform，WT）  逆小波变换（Inverse Wavelet Transform，IWT）
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        #base_conv 是一个基础的深度可分卷积 groups=in_channels 意味着每个输入通道独立处理
        # self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
        #                            groups=in_channels, bias=bias)  #原本
        self.base_conv = DeformConv2d(in_channels, in_channels, kernel_size, padding=2, stride=1, dilation=1,
                                   groups=in_channels, bias=bias)

        #对卷积输出进行缩放操作
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])


        self.wavelet_convs = nn.ModuleList(
           [DeformConv2d(in_channels * 4, in_channels * 4, kernel_size, padding=2, stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        #wavelet_scale 是每个小波层的缩放模块，用于缩放小波变换后的结果
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        #x_ll_in_levels 和 x_h_in_levels 用于保存每一层的小波低频和高频部分
        x_ll_in_levels = [] #用于存储每一层分解后的低频分量
        x_h_in_levels = [] #用于存储每一层分解后的高频分量
        shapes_in_levels = [] #用于记录每一层输入张量的形状，以便后续重构时对齐原尺寸

        curr_x_ll = x
        #遍历每一层的小波变换
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape) #输入 curr_x_ll 的形状记录到 shapes_in_levels
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0): #如果图像尺寸为奇数（无法均匀分解），则对宽高方向进行边界填充
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll) #对输入进行小波分解
            curr_x_ll = curr_x[:, :, 0, :, :]
            #在每一层，将当前输入 curr_x_ll 进行小波变换，并保存低频部分 curr_x_ll
            shape_x = curr_x.shape
            #将小波变换的结果通过卷积和尺度模块进一步处理，得到更高层次的特征表示
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4]) #4 个子带：低频 LL 和 3 个高频 LH, HL, HH
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)) #进行卷积操作 self.wavelet_convs[i] 和缩放操作 self.wavelet_scale[i]
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])   #分别提取低频和高频分量，存储到 x_ll_in_levels 和 x_h_in_levels
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])   #高频分量


        #逆小波
        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            #取出当前层的小波低频分量 curr_x_ll 和高频分量 curr_x_h，以及记录的输入形状 curr_shape
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll  #当前层的低频分量与上一层逆变换结果 next_x_ll 相加

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2) #将低频分量 curr_x_ll 和高频分量 curr_x_h 拼接
            next_x_ll = self.iwt_function(curr_x) #使用 iwt_function 逆小波变换

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]] #裁剪恢复后的张量到记录的原始尺寸

        x_tag = next_x_ll #记录了经过多层小波分解和重构后的张量
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        #将经过小波变换的特征图和基础卷积的输出结合起来
        x = x + x_tag  #将卷积特征与小波特征 x_tag 相加，融合多分辨率和卷积学习的特征

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class LayerScale(nn.Module):
    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)


class TransformerStage(nn.Module):
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, 
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList()
        for _ in range(depths):
            if use_lpu:
                self.local_perception_units.append(nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed))
            else: 
                self.local_perception_units.append(nn.Identity())
       
        for i in range(depths):
            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, heads, nat_ksize, attn_drop=attn_drop, proj_drop=proj_drop)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)
        for d in range(self.depths):
            
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class LinearPatchExpand2D(nn.Module):
    def __init__(self, dim, scale_factor=2, norm_layer=LayerNormProxy):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.output_dim = dim // scale_factor if scale_factor == 2 else dim

        self.expand = nn.Linear(dim, scale_factor*dim if scale_factor==2 else (scale_factor**2)*dim, bias=False) if scale_factor > 1 else nn.Identity()
        self.norm = norm_layer(dim // scale_factor if scale_factor==2 else dim) 

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print(f"X1 is {x.shape}")# ([18, 1024, 7, 7])
        x = x.flatten(2).permute(0, 2, 1)
        # print(f"X2 is {x.shape}") #([18, 49, 1024])
        x = self.expand(x)
        # print(f"X3 is {x.shape}") # ([18, 49, 2048])
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        assert L == H * W, "input feature has wrong size"
        # print(f"X4 is {x.shape}") #([18, 49, 2048])
        x = x.view(B, H, W, C)
        # print(f"X5 is {x.shape}") #([18, 7, 7, 2048])
        #'b h w (p1 p2 c)' 表示输入张量的形状：   'b (h p1) (w p2) c' 表示输出张量的目标形状：
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.scale_factor, p2=self.scale_factor, c=self.output_dim)
        # print(f"X6 is {x.shape}") #([18, 14, 14, 512])
        x = x.reshape(B, H*self.scale_factor, W*self.scale_factor, self.output_dim) # BxHxWxC
        # print(f"X7 is {x.shape}") #([18, 14, 14, 512])
        x = x.permute(0, 3, 1, 2) # BxCxHxW
        # print(f"X8 is {x.shape}") #([18, 512, 14, 14])
        x= self.norm(x)
        # print(f"X9 is {x.shape}")  #([18, 512, 14, 14])

        return x


class FeatureAggregation(nn.Module):
    def __init__(self, msc_layer, mfms_attention_layer):
        super(FeatureAggregation, self).__init__()
        self.msc_layer = msc_layer
        self.mfms_attention_layer = mfms_attention_layer

    def forward(self, x):
        x1 = self.msc_layer(x)
        x2 = self.mfms_attention_layer(x)
        return x1 + x2

class FeatureAggregation2(nn.Module):
    def __init__(self, wtconv_layer, sspcab_attention_layer):
        super(FeatureAggregation2, self).__init__()
        self.wtconv_layer = wtconv_layer
        self.sspcab_attention_layer = sspcab_attention_layer

    def forward(self, x):
        x1 = self.wtconv_layer(x)
        x2 = self.sspcab_attention_layer(x)
        return x1 + x2

# class FeatureAggregation3(nn.Module):
#     def __init__(self, up_proj, ucb_layer):
#         super(FeatureAggregation3, self).__init__()
#         self.up_proj = up_proj
#         self.ucb_layer = ucb_layer
#
#     def forward(self, x):
#         x1 = self.up_proj(x)
#         x2 = self.ucb_layer(x)
#         return x1 + x2


class WDMCADSys2D(nn.Module):
    def __init__(self, img_size=224, patch_size=4, expansion=4, num_classes=9,
                 dim_stem=96, dims=[96, 192, 384, 768], 
                 depths_Encoder=[2, 2, 6, 2], depths_Decoder=[2, 2, 6, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1],
                 offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 nat_ksizes=[3,3,3,3],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 encoder_pos_layers=[0, 1, 2],
                 decoder_pos_layers=[1, 2, 3],
                 feature_sizes=[56, 28, 14, 7],
                 deep_supervision=False,

                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.msc_layers = nn.ModuleList([
            MSCLayer(in_channels=dims[i - 1], out_channels=dims[i - 1])
            for i in range(1, len(dims))
        ])

        self.attention_layers = nn.ModuleList([
            AttentionBlock(in_channels=dims[i - 1])
            for i in range(1, len(dims))
        ])

        self.wdconv_layers = nn.ModuleList([
            WDConv2d(in_channels=dims[i - 1], out_channels=dims[i - 1])
            for i in range(1, len(dims))
        ])

        self.mab_attention_layers = nn.ModuleList([
            MAB(channels=dims[i - 1])
            for i in range(1, len(dims))
        ])

        self.feature_aggregation_layers = nn.ModuleList([
            FeatureAggregation(msc_layer, attention_layer)
            for msc_layer, attention_layer in zip(self.msc_layers, self.attention_layers)
        ])

        self.feature_aggregation_layers2 = nn.ModuleList([
            FeatureAggregation2(wdconv_layer, mab_attention_layer)
            for wdconv_layer, mab_attention_layer in zip(self.wdconv_layers, self.mab_attention_layers)
        ])
        self.patch_proj = nn.Sequential(
            DeformConv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            DeformConv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            DeformConv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )   #原本

        img_size = img_size // patch_size
        
        ################ encoder ################
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_Encoder))]
        # print(len(dpr))
        self.stages = nn.ModuleList()
        self.deform_pos_encoder = nn.ModuleList()
        use_cpe_encoder = encoder_pos_layers[0] != -1
        
        for i in range(len(depths_Encoder)):
            use_cpe_encoder = use_cpe_encoder and (i in encoder_pos_layers)

            if use_cpe_encoder:
                self.deform_pos_encoder.append(DePE(dims[i], dims[i], conv_op=DeformConv2d, groups=dims[i])) #原本
                # self.deform_pos_decoder.append(DePE(dims[i], dims[i], conv_op=WDConv2d, groups=dims[i]))
            else:
                self.deform_pos_encoder.append(nn.Identity())
            
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dims[i], dims[i], depths_Encoder[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths_Encoder[:i]):sum(depths_Encoder[:i + 1])], use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i],qna_activation,
                    layer_scale_values[i], use_lpus[i], log_cpb[i]
                )
            )
            if i < 3:
                img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.cls_norm = LayerNormProxy(dims[-1]) 

        self.stages_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.deform_pos_decoder = nn.ModuleList()
        self.ds_projs = nn.ModuleList()
        use_cpe_decoder = decoder_pos_layers[0] != -1
   
        for i in range(1, len(depths_Decoder)):
            idx = len(depths_Decoder)-1-i

            use_cpe_decoder = use_cpe_decoder and (i in decoder_pos_layers)

            if use_cpe_decoder:
                self.deform_pos_decoder.append(DePE(dims[idx], dims[idx], conv_op=DeformConv2d, groups=dims[idx])) #原本
                # self.deform_pos_decoder.append(DePE(dims[idx], dims[idx], conv_op=WDConv2d, groups=dims[idx]))
                # self.deform_pos_decoder.append(DePE(dims[idx], dims[idx], conv_op=KAN_Convolution, groups=dims[idx]))
            else:
                self.deform_pos_decoder.append(nn.Identity())

            img_size = img_size * 2

            self.stages_up.append(
                TransformerStage(
                    img_size, window_sizes[idx], ns_per_pts[idx],
                    dims[idx], dims[idx], depths_Decoder[idx],
                    stage_spec[idx], groups[idx], use_pes[idx],
                    sr_ratios[idx], heads[idx], heads_q[idx], strides[idx],
                    offset_range_factor[idx],
                    dwc_pes[idx], no_offs[idx], fixed_pes[idx],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths_Decoder[:idx]):sum(depths_Decoder[:idx + 1])], use_dwc_mlps[i],
                    ksizes[idx], nat_ksizes[idx], ksize_qnas[idx], nqs[idx],qna_activation,
                    layer_scale_values[idx], use_lpus[idx], log_cpb[idx]
                )
            )

            if deep_supervision:
                self.ds_projs.append(nn.Sequential(
                    LayerNormProxy(dims[idx]),
                    LinearPatchExpand2D(dims[idx], scale_factor=4),
                    nn.Conv2d(in_channels=dims[idx],out_channels=self.num_classes,kernel_size=1,bias=False)
                ))
            else:
                self.ds_projs.append(nn.Identity())
       
            self.concat_back_dim.append(nn.Conv2d(dims[idx]*2, dims[idx], 1, 1, 0))



        self.up_projs = nn.ModuleList()
        for i in range(len(depths_Decoder)):
            idx = len(depths_Decoder)-1-i
            # print(groups[idx])
            scale_factor = 2 if i < len(depths_Decoder) - 1 else 4
            self.up_projs.append(
                LinearPatchExpand2D(dims[idx], scale_factor=scale_factor),
            )

        # self.feature_aggregation_layers3 = nn.ModuleList([
        #     FeatureAggregation3(up_proj, ucb_layer)
        #     for up_proj, ucb_layer in zip(self.up_projs, self.ucb_layers)
        # ])

        self.norm_up= LayerNormProxy(dims[0])
        self.output = nn.Conv2d(in_channels=dims[0],out_channels=self.num_classes,kernel_size=1,bias=False)

        self.lower_lr_kvs = lower_lr_kvs
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict):
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            if "patch_proj" in state_key:
                new_state_dict[state_key] = state_value
            elif "down_projs" in state_key:
                new_state_dict[state_key] = state_value
            elif "cls_norm" in state_key:
                new_state_dict[state_key] = state_value
            elif "stages" in state_key and state_key in self.state_dict().keys():
                if self.state_dict()[state_key].shape == state_value.shape:
                    new_state_dict[state_key] = state_value
                    tmp = state_key.split(".")
                    num_depth = int(tmp[1])
                    if num_depth < 3:
                        new_state_key = ["stages_up", str(2 - num_depth)] + tmp[2:]
                        new_state_key = ".".join(new_state_key)

                        try:
                            if self.state_dict()[new_state_key].shape == state_value.shape:
                                new_state_dict[new_state_key] = state_value
                            else:
                                pass
                        except:
                            pass
                else:
                    pass
            else:
                pass

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    def forward_encoder(self, x):

        # print(f"X0 is {x.shape}") #([18, 3, 224, 224])
        x = self.patch_proj(x)
        # print(f"X1 is {x.shape}")  ##([18, 128, 56, 56])
        x_downsample = []
        for i in range(len(self.stages)):
            # print(f"X2 is {x.shape}") #([18, 128, 56, 56])
            x_downsample.append(x)
            # print(f"X3 is {x.shape}") #([18, 128, 56, 56])
            # x = self.deform_pos_encoder[i](x)
            # print(f"X4 is {x.shape}") #([18, 128, 56, 56])
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
        x = self.cls_norm(x)
        # print(x.shape)
        return x, x_downsample
    
    def forward_decoder(self, x, x_downsample):
        seg_outputs = []
        for i in range(len(self.stages)):
            if i == 0:

                x = self.up_projs[i](x)

            else:

                x = self.feature_aggregation_layers2[3 - i](x)

                x = torch.cat([x, x_downsample[len(self.stages)-1-i]], 1)

                x = self.concat_back_dim[i-1](x)
                x = self.feature_aggregation_layers[3 - i](x)

                # x = self.stages_up[i-1](x)

             
                if self.deep_supervision and (i < 3):
                    seg_outputs.append(self.ds_projs[i-1](x)) 
               
                if i < 3:
                    # print(f"X3 is {x.shape}") #([18, 512, 14, 14])
                    x = self.up_projs[i](x)

                 
        x = self.norm_up(x)  # B L C
        x = self.up_projs[-1](x)
        x = self.output(x)

        if len(seg_outputs) > 0:
            # print(True)
            x += F.interpolate(F.interpolate(seg_outputs[-2], scale_factor=2, mode='bilinear') + seg_outputs[-1],
                               scale_factor=2, mode='bilinear')

        return x
    
    def forward(self, x):
        x, x_downsample = self.forward_encoder(x)
        # print(self.forward_decoder(x, x_downsample))
        x = self.forward_decoder(x, x_downsample)

        return x
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table', 'deform_pos_encoder', 'deform_pos_decoder'}