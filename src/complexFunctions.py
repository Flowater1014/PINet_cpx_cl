#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
from torch.nn import Sigmoid
import torch

def complex_mul(A, B):
    '''
        Performs the matrix product between two complex matricess
    '''

    outp_real = torch.mul(A.real, B.real) - torch.mul(A.imag, B.imag)
    outp_imag = torch.mul(A.real, B.imag) + torch.mul(A.imag, B.real)
    
    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)

def complex_matmul(A, B):
    '''
        Performs the matrix product between two complex matricess
    '''

    outp_real = torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
    outp_imag = torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real)
    
    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)

def complex_avg_pool2d(input, *args, **kwargs):
    '''
    Perform complex average pooling.
    '''    
    absolute_value_real = avg_pool2d(input.real, *args, **kwargs)
    absolute_value_imag =  avg_pool2d(input.imag, *args, **kwargs)    
    
    return absolute_value_real.type(torch.complex64)+1j*absolute_value_imag.type(torch.complex64)

def complex_adaptive_avg_pool2d(input, output_size):
    absolute_value_real = adaptive_avg_pool2d(input.real, output_size)
    absolute_value_imag = adaptive_avg_pool2d(input.imag, output_size)
    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(torch.complex64)

def complex_max_pool2d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    absolute_value, indices =  max_pool2d(
                               input.abs(), 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding, 
                               dilation = dilation,
                               ceil_mode = ceil_mode, 
                               return_indices = True
                            )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresonding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(input.imag,input.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value \
           * (torch.cos(angle).type(torch.complex64)+1j*torch.sin(angle).type(torch.complex64))

    
def complex_adaptive_max_pool2d(input, output_size, return_indices=False):
    absolute_value, indices = adaptive_max_pool2d(
        input.abs(), output_size, return_indices=True)
    absolute_value = absolute_value.type(torch.complex64)
    angle = torch.atan2(input.imag, input.real)
    angle = _retrieve_elements_from_indices(angle, indices)
    result = absolute_value * (torch.cos(angle).type(torch.complex64) + 1j * torch.sin(angle).type(torch.complex64))
    if return_indices:
        return result, indices
    return result


def complex_normalize(input):
    '''
    Perform complex normalization
    '''
    real_value, imag_value = input.real, input.imag
    real_norm = (real_value - real_value.mean()) / real_value.std()
    imag_norm = (imag_value - imag_value.mean()) / imag_value.std()
    
    return real_norm.type(torch.complex64) + 1j*imag_norm.type(torch.complex64)

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

def complex_sigmoid(input):
    sig = Sigmoid()
    return sig(input.real).type(torch.complex64)+1j*sig(input.imag).type(torch.complex64)

def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output

def complex_upsample(input, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, recompute_scale_factor=None):
    '''
        Performs upsampling by separately interpolating the real and imaginary part and recombining
    '''
    outp_real = interpolate(input.real,  size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    outp_imag = interpolate(input.imag,  size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    
    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)

def complex_upsample2(input, size=None, scale_factor=None, mode='nearest',
                      align_corners=None, recompute_scale_factor=None):
    '''
        Performs upsampling by separately interpolating the amplitude and phase part and recombining
    '''
    outp_abs = interpolate(
        input.abs(),
        size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor
    )

    angle = torch.atan2(input.imag, input.real)
    outp_angle = interpolate(
        angle,
        size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor
    )

    # ✅ 用 outp_angle 来重建
    return outp_abs * (
        torch.cos(outp_angle).type(torch.complex64) +
        1j * torch.sin(outp_angle).type(torch.complex64)
    )





def complex_dropout(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part, 
    # this not a clean solution!
    #mask = torch.ones_like(input).type(torch.float32)
    mask = torch.ones(*input.shape, dtype = torch.float32)
    mask = dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    return mask*input


def complex_dropout2d(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part, 
    # this not a clean solution!
    mask = torch.ones(*input.shape, dtype = torch.float32)
    mask = dropout2d(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    return mask*input
