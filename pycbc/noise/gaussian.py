# Copyright (C) 2012  Alex Nitz
#
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""This module contains functions to generate gaussian noise colored with a 
noise spectrum. 
"""
from pycbc.types import TimeSeries, zeros, Array
from pycbc.types import complex_same_precision_as, FrequencySeries
from lalsimulation import SimNoise
import lal
import numpy.random

def frequency_noise_from_psd(psd, seed = None):
    """ Create noise with a given psd.
    
    Return noise coloured with the given psd. The returned noise 
    FrequencySeries has the same length and frequency step as the given psd. 
    Note that if unique noise is desired a unique seed should be provided.

    Parameters
    ----------
    psd : FrequencySeries
        The noise weighting to color the noise.
    seed : {0, int} or None
        The seed to generate the noise. If None specified,
        the seed will not be reset.
        
    Returns
    --------
    noise : FrequencySeriesSeries
        A FrequencySeries containing gaussian noise colored by the given psd. 
    """
    sigma = 0.5 * (psd / psd.delta_f) ** (0.5) 
    if seed is not None:
        numpy.random.seed(seed)
    sigma = sigma.numpy()
    dtype = complex_same_precision_as(psd)
    
    not_zero = (sigma != 0)
    
    sigma_red = sigma[not_zero]
    noise_re = numpy.random.normal(0, sigma_red)
    noise_co = numpy.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co
    
    noise = numpy.zeros(len(sigma), dtype=dtype)
    noise[not_zero] = noise_red
    
    return FrequencySeries(noise,
                           delta_f=psd.delta_f,
                           dtype=dtype)
    
    
   

def noise_from_psd(length, delta_t, psd, seed=0):
    """ Create noise with a given psd.
    
    Return noise with a given psd. Note that if unique noise is desired 
    a unique seed should be provided.

    Parameters
    ----------
    length : int
        The length of noise to generate in samples.
    delta_t : float
        The time step of the noise. 
    psd : FrequencySeries
        The noise weighting to color the noise.
    seed : {0, int}
        The seed to generate the noise. 
        
    Returns
    --------
    noise : TimeSeries
        A TimeSeries containing gaussian noise colored by the given psd. 
    """
    noise_ts = TimeSeries(zeros(length), delta_t=delta_t)
    
    randomness = lal.gsl_rng("ranlux", seed)
    
    N = int (1.0 / delta_t / psd.delta_f)
    n = N/2+1
    stride = N/2
    
    if n > len(psd):
        raise ValueError("PSD not compatible with requested delta_t")
        
    psd = (psd[0:n]).lal()
    psd.data.data[n-1] = 0 
    
    segment = TimeSeries(zeros(N), delta_t=delta_t).lal()
    length_generated = 0 
  
    SimNoise(segment, 0, psd, randomness) 
    while (length_generated < length):
        if (length_generated + stride) < length:
            noise_ts.data[length_generated:length_generated+stride] = segment.data.data[0:stride]
        else: 
            noise_ts.data[length_generated:length] = segment.data.data[0:length-length_generated]
    
        length_generated += stride
        SimNoise(segment, stride, psd, randomness)
        
    return noise_ts
   
__all__ = ['noise_from_psd', 'frequency_noise_from_psd']
