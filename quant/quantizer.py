import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False,
                 two_part:bool = False, split_quantization: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.register_buffer('inited', torch.zeros(1))
        self.channel_wise = channel_wise
        self.is_act = is_act
        self.store_input = None
        self.two_part = two_part
        self.split_quantization = split_quantization
        self.negative_mask = None

    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor, round_way='round', index=None):

        if self.inited == 0:
            if not self.two_part:
                delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
            else:

                indices_list = []
                for ii in range(x.shape[0]):

                    per = 99
                    vector = x[ii].flatten().detach().cpu().numpy()
                    vector = vector[vector > 0]
                    threshold_pos = np.percentile(vector, per)

                    vector = x[ii].flatten().detach().cpu().numpy()
                    vector = vector[vector < 0]
                    threshold_neg = np.percentile(vector, 100-per)

                    indices_gt = torch.nonzero(x[ii] > threshold_pos)
                    indices_lt = torch.nonzero(x[ii] < threshold_neg)
                    indices = torch.unique(torch.cat((indices_gt, indices_lt), dim=0))
                    indices_list.append(indices)

                count_dict = Counter()
                for item in indices_list:
                    count_dict.update(item.tolist())

                best_top_number = x.shape[0] // 20
                print('best_top_number', best_top_number)
                # _, top_values_tensor = torch.topk(outliers_count, best_top_number)
                top_ = count_dict.most_common(best_top_number)
                top_values = [value for value, count in top_]
                top_values_tensor = torch.tensor(top_values).cuda()
                self.unique_top_indices = top_values_tensor
                tmp = []
                for i in range(x.shape[1]):
                    if i not in self.unique_top_indices:
                        tmp.append(i)
                self.unique_nottop_indices = torch.tensor(tmp).cuda()
                x_1 = x[:, self.unique_top_indices]
                x_2 = x[:, self.unique_nottop_indices]

                delta, zero_point = self.init_quantization_scale(x_2, self.channel_wise)
                self.delta_2, self.zero_point_2 = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()

                delta, zero_point = self.init_quantization_scale(x_1, self.channel_wise)
                self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()


            self.inited.fill_(1)

        if not self.two_part:
            # start quantization
            if index is None:
                if round_way == 'round':
                    x_int = torch.round(x / self.delta) + self.zero_point
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point) * self.delta
                elif round_way == 'floor':
                    x_int = torch.floor(x / self.delta) + self.zero_point
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point) * self.delta
                elif round_way == 'ceil':
                    x_int = torch.ceil(x / self.delta) + self.zero_point
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point) * self.delta
                else:
                    raise Exception
            else:
                if round_way == 'round':
                    x_int = torch.round(x / self.delta[index]) + self.zero_point[index]
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
                elif round_way == 'floor':
                    x_int = torch.floor(x / self.delta[index]) + self.zero_point[index]
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
                elif round_way == 'ceil':
                    x_int = torch.ceil(x / self.delta[index]) + self.zero_point[index]
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
                else:
                    raise Exception
        else:
            if index is None:
                # start quantization
                x_1 = x[:, self.unique_top_indices]
                x_2 = x[:, self.unique_nottop_indices]

                if self.split_quantization:
                    self.negative_mask = torch.sign(x_1)

                if round_way == 'round':
                    if not self.split_quantization:

                        x_int_1 = torch.round(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                    else:

                        x_1 = self.negative_mask * x_1
                        x_int_1 = torch.round(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                        x_dequant_1 = self.negative_mask * x_dequant_1

                    x_int_2 = torch.round(x_2 / self.delta_2) + self.zero_point_2
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
                elif round_way == 'floor':
                    if not self.split_quantization:
                        x_int_1 = torch.floor(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                    else:
                        x_1 = self.negative_mask * x_1
                        x_int_1 = torch.floor(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                        x_dequant_1 = self.negative_mask * x_dequant_1

                    x_int_2 = torch.floor(x_2 / self.delta_2) + self.zero_point_2
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
                elif round_way == 'ceil':
                    if not self.split_quantization:
                        x_int_1 = torch.ceil(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                    else:
                        x_1 = self.negative_mask * x_1
                        x_int_1 = torch.ceil(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                        x_dequant_1 = self.negative_mask * x_dequant_1


                    x_int_2 = torch.ceil(x_2 / self.delta_2) + self.zero_point_2
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
                else:
                    raise Exception

                x_dequant = torch.zeros(x.shape).cuda()
                x_dequant[:, self.unique_top_indices] = x_dequant_1
                x_dequant[:, self.unique_nottop_indices] = x_dequant_2
            else:
                # start quantization
                x_1 = x[self.unique_top_indices]
                x_2 = x[self.unique_nottop_indices]

                if round_way == 'round':
                    if not self.split_quantization:
                        x_int_1 = torch.round(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                    else:

                        x_1 = self.negative_mask[index] * x_1
                        x_int_1 = torch.round(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, (self.n_levels//2) - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                        x_dequant_1 = self.negative_mask[index] * x_dequant_1

                    x_int_2 = torch.round(x_2 / self.delta_2[index]) + self.zero_point_2[index]
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2[index]) * self.delta_2[index]
                elif round_way == 'floor':
                    if not self.split_quantization:
                        x_int_1 = torch.floor(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                    else:
                        x_1 = self.negative_mask[index] * x_1
                        x_int_1 = torch.floor(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, (self.n_levels//2) - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                        x_dequant_1 = self.negative_mask[index] * x_dequant_1

                    x_int_2 = torch.floor(x_2 / self.delta_2[index]) + self.zero_point_2[index]
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2[index]) * self.delta_2[index]
                elif round_way == 'ceil':
                    if not self.split_quantization:
                        x_int_1 = torch.ceil(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                    else:
                        x_1 = self.negative_mask[index] * x_1
                        x_int_1 = torch.ceil(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, (self.n_levels//2) - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                        x_dequant_1 = self.negative_mask[index] * x_dequant_1

                    x_int_2 = torch.ceil(x_2 / self.delta_2[index]) + self.zero_point_2[index]
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2[index]) * self.delta_2[index]
                else:
                    raise Exception
                x_dequant = torch.zeros(x.shape).cuda()
                x_dequant[self.unique_top_indices] = x_dequant_1
                x_dequant[self.unique_nottop_indices] = x_dequant_2
        return x_dequant


    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if self.is_act:
                # n_channels = x_clone.shape[-1] if len(x.shape) == 3
                if len(x.shape) == 3: # torch.Size([64, 197, 384])
                    n_channels = x_clone.shape[-1]
                elif len(x.shape) == 4: # torch.Size([64, 3, 224, 224]) or q k v to-do:
                    n_channels = x_clone.shape[1] # quantization dim=3
                elif len(x.shape) == 2: # torch.Size([64, 384])
                    n_channels = x_clone.shape[1] # quantization dim=384
                else:
                    raise NotImplementedError

                if len(x.shape) == 4: 
                    # x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:# 
                    x_max = x_clone.abs().max(dim=0)[0]
                elif len(x.shape) == 3: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,...], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False)
                if len(x.shape) == 4:

                    delta = delta.reshape(1, -1, 1, 1)
                    zero_point = zero_point.reshape(1, -1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.reshape(1, -1)
                    zero_point = zero_point.reshape(1, -1)
                elif len(x.shape) == 3:
                    delta = delta.reshape(1, 1, -1)
                    zero_point = zero_point.reshape(1, 1, -1)
                else:
                    raise NotImplementedError
            else: 
                n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c],
                                                                               channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c],
                                                                               channel_wise=False)
                if len(x.shape) == 4:
                    delta = delta.reshape(-1, 1, 1, 1)
                    zero_point = zero_point.reshape(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.reshape(-1, 1)
                    zero_point = zero_point.reshape(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.reshape(1, 1, -1)
                    zero_point = zero_point.reshape(1, 1, -1)
                else:
                    raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            if self.is_act:
                search_range = [0.999, 0.9999, 0.99999]
            else:
                search_range = [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1]

            if not self.split_quantization:
                for pct in search_range:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                x_clone = torch.abs(x_clone)
                for pct in search_range:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** (self.n_bits-1) - 1)
                        zero_point = (- new_min / delta).round()


        return delta, zero_point


    def quantize(self, x, max, min):
        if not self.split_quantization:
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            # we assume weight quantization is always signed
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
            return x_float_q
        else:
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            # we assume weight quantization is always signed
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, (self.n_levels//2) - 1)
            x_float_q = (x_quant - zero_point) * delta
            return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise


    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0

        # x_int = torch.round(-1 * (x / delta).log2())
        # mask = x_int >= self.n_levels
        # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # x_float_q = 2 ** (-1 * x_quant) * delta
        # x_float_q[mask] = 0
        #
        return x_float_q
