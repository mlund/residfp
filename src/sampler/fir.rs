// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! FIR filter coefficients and initialization.
//!
//! Implements Kaiser-windowed sinc filter design for high-quality audio resampling.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use libm::F64Ext;

use super::{SamplingMethod, FIR_RES_FAST, FIR_RES_INTERPOLATE, FIR_SHIFT};

/// Default passband limit for resampling (Hz)
pub const DEFAULT_PASS_FREQ: f64 = 20000.0;

/// FIR filter coefficients.
#[cfg(feature = "alloc")]
#[derive(Clone, Default)]
pub struct Fir {
    pub data: Vec<i16>,
    pub n: i32,
    pub res: i32,
}

/// Compute the 0th order modified Bessel function of the first kind.
pub fn i0(x: f64) -> f64 {
    const I0E: f64 = 1e-6;
    let halfx = x / 2.0;
    let mut sum = 1.0;
    let mut u = 1.0;
    let mut n = 1;
    loop {
        let temp = halfx / n as f64;
        n += 1;
        u *= temp * temp;
        sum += u;
        if u < I0E * sum {
            break;
        }
    }
    sum
}

/// Square root with libm fallback for no_std.
#[cfg(feature = "std")]
pub fn sqrt_compat(value: f64) -> f64 {
    value.sqrt()
}

#[cfg(not(feature = "std"))]
pub fn sqrt_compat(value: f64) -> f64 {
    libm::sqrt(value)
}

/// Initialize FIR filter for single-pass resampling.
#[cfg(feature = "alloc")]
pub fn init_fir(
    fir: &mut Fir,
    sampling_method: SamplingMethod,
    clock_freq: f64,
    sample_freq: f64,
    mut pass_freq: f64,
    filter_scale: f64,
) {
    let pi = core::f64::consts::PI;
    let samples_per_cycle = sample_freq / clock_freq;
    let cycles_per_sample = clock_freq / sample_freq;

    // The default passband limit is 0.9*sample_freq/2 for sample
    // frequencies below ~44.1kHz, and 20kHz for higher sample frequencies.
    if pass_freq < 0.0 {
        pass_freq = DEFAULT_PASS_FREQ;
        if 2.0 * pass_freq / sample_freq >= 0.9 {
            pass_freq = 0.9 * sample_freq / 2.0;
        }
    }

    // 16 bits -> -96dB stopband attenuation.
    let atten = -20.0_f64 * (1.0 / (1_i32 << 16) as f64).log10();
    // A fraction of the bandwidth is allocated to the transition band,
    let dw = (1.0_f64 - 2.0 * pass_freq / sample_freq) * pi;
    // The cutoff frequency is midway through the transition band.
    let wc = (2.0_f64 * pass_freq / sample_freq + 1.0) * pi / 2.0;

    // For calculation of beta and N see the reference for the kaiserord
    // function in the MATLAB Signal Processing Toolbox:
    // http://www.mathworks.com/access/helpdesk/help/toolbox/signal/kaiserord.html
    let beta = 0.1102_f64 * (atten - 8.7);
    let io_beta = i0(beta);

    // The filter order will maximally be 124 with the current constraints.
    // N >= (96.33 - 7.95)/(2.285*0.1*pi) -> N >= 123
    // The filter order is equal to the number of zero crossings, i.e.
    // it should be an even number (sinc is symmetric about x = 0).
    let mut n_cap = ((atten - 7.95) / (2.285 * dw) + 0.5) as i32;
    n_cap += n_cap & 1;

    // The filter length is equal to the filter order + 1.
    // The filter length must be an odd number (sinc is symmetric about x = 0).
    fir.n = (n_cap as f64 * cycles_per_sample) as i32 + 1;
    fir.n |= 1;

    // We clamp the filter table resolution to 2^n, making the fixpoint
    // sample_offset a whole multiple of the filter table resolution.
    let res = if sampling_method == SamplingMethod::Resample {
        FIR_RES_INTERPOLATE
    } else {
        FIR_RES_FAST
    };
    let n = ((res as f64 / cycles_per_sample).ln() / 2.0_f64.ln()).ceil() as i32;
    fir.res = 1 << n;

    fir.data.clear();
    fir.data.resize((fir.n * fir.res) as usize, 0);

    // Calculate fir_RES FIR tables for linear interpolation.
    for i in 0..fir.res {
        let fir_offset = i * fir.n + fir.n / 2;
        let j_offset = i as f64 / fir.res as f64;
        // Calculate FIR table. This is the sinc function, weighted by the
        // Kaiser window.
        let fir_n_div2 = fir.n / 2;
        for j in -fir_n_div2..=fir_n_div2 {
            let jx = j as f64 - j_offset;
            let wt = wc * jx / cycles_per_sample;
            let temp = jx / fir_n_div2 as f64;
            let kaiser = if temp.abs() <= 1.0 {
                i0(beta * sqrt_compat(1.0 - temp * temp)) / io_beta
            } else {
                0_f64
            };
            let sincwt = if wt.abs() >= 1e-6 { wt.sin() / wt } else { 1.0 };
            let val = (1_i32 << FIR_SHIFT) as f64 * filter_scale * samples_per_cycle * wc / pi
                * sincwt
                * kaiser;
            fir.data[(fir_offset + j) as usize] = (val + 0.5) as i16;
        }
    }
}
