// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

// Allow cast_lossless: intentional i16->i32 casts for audio sample processing
#![allow(clippy::cast_lossless)]
// Allow cast_ptr_alignment: SIMD pointer casts are aligned by construction
#![allow(clippy::cast_ptr_alignment)]

use core::f64;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::synth::Synth;
use super::SamplingError;
#[cfg(not(feature = "std"))]
use libm::F64Ext;

use wide::{i16x16, i32x8};

// Resampling constants.
// The error in interpolated lookup is bounded by 1.234/L^2,
// while the error in non-interpolated lookup is bounded by
// 0.7854/L + 0.4113/L^2, see
// http://www-ccrma.stanford.edu/~jos/resample/Choice_Table_Size.html
// For a resolution of 16 bits this yields L >= 285 and L >= 51473,
// respectively.
const FIR_RES_FAST: i32 = 51473;
const FIR_RES_INTERPOLATE: i32 = 285;
const FIR_SHIFT: i32 = 15;
const RING_SIZE: usize = 16384;
const RING_MASK: usize = RING_SIZE - 1;

const FIXP_SHIFT: i32 = 16;
const FIXP_MASK: i32 = 0xffff;

/// Soft clipping threshold - values below pass unchanged
const SOFT_CLIP_THRESHOLD: i32 = 28000;

/// Default passband limit for resampling (Hz)
const DEFAULT_PASS_FREQ: f64 = 20000.0;

/// Padé tanh approximation (5th order), accurate for |x| < 3.
#[inline]
fn tanh_pade(x: f64) -> f64 {
    if x.abs() < 3.0 {
        let x2 = x * x;
        let num = x * (945.0 + x2 * (105.0 + x2));
        let den = 945.0 + x2 * (420.0 + x2 * 15.0);
        num / den
    } else {
        // Beyond approximation range, use sign
        if x > 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// Soft clip positive value using tanh curve above threshold.
/// Creates smooth saturation instead of hard clipping.
#[inline]
fn soft_clip_positive(x: i32, max_val: i32) -> i32 {
    if x < SOFT_CLIP_THRESHOLD {
        return x;
    }

    let max_f = max_val as f64;
    let t = SOFT_CLIP_THRESHOLD as f64 / max_f;
    let a = 1.0 - t;
    let b = 1.0 / a;

    let value = (x - SOFT_CLIP_THRESHOLD) as f64 / max_f;
    let result = t + a * tanh_pade(b * value);
    (result * max_f) as i32
}

/// Soft clip into 16-bit range [i16::MIN, i16::MAX].
/// Uses tanh curve for smooth saturation above threshold.
#[inline]
pub fn soft_clip(x: i32) -> i16 {
    if x < 0 {
        // Handle i32::MIN overflow case
        let abs_x = if x == i32::MIN { i32::MAX } else { -x };
        // Negate as i32 first to avoid i16 overflow
        // Note: i16::MIN magnitude is 32768, one more than i16::MAX
        (-soft_clip_positive(abs_x, (i16::MIN as i32).abs())) as i16
    } else {
        soft_clip_positive(x, i16::MAX as i32) as i16
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum SamplingMethod {
    Fast,
    Interpolate,
    #[cfg(feature = "alloc")]
    Resample,
    #[cfg(feature = "alloc")]
    ResampleFast,
    #[cfg(feature = "alloc")]
    ResampleTwoPass,
}

#[derive(Clone)]
struct Fir {
    data: Vec<i16>,
    n: i32,
    res: i32,
}

/// Ring buffer size for two-pass resampler.
/// Smaller than single-pass (2048 vs 16384) since each pass handles less decimation.
const TWO_PASS_RING_SIZE: usize = 2048;
const TWO_PASS_RING_MASK: usize = TWO_PASS_RING_SIZE - 1;

/// Fixed-point scale for two-pass resampler.
/// Uses 10 bits (vs 16 for single-pass) to match libresidfp's precision tradeoffs.
const TWO_PASS_FIXP_SHIFT: i32 = 10;
const TWO_PASS_FIXP_SCALE: i32 = 1 << TWO_PASS_FIXP_SHIFT;
const TWO_PASS_FIXP_MASK: i32 = TWO_PASS_FIXP_SCALE - 1;

/// Passband limit for high sample rates (>44kHz).
/// 20kHz covers full audible range while leaving transition band headroom.
const TWO_PASS_DEFAULT_PASSBAND: f64 = 20000.0;

/// Two-pass sinc resampler using chained FIR filters.
///
/// Chains two FIR filters through an intermediate frequency for improved
/// computational efficiency. The intermediate frequency is calculated using
/// Laurent Ganier's formula to minimize total filter order.
#[cfg(feature = "alloc")]
#[derive(Clone)]
struct TwoPassResampler {
    /// Pass 1 FIR: clock_freq → intermediate_freq
    fir1: Fir,
    /// Pass 2 FIR: intermediate_freq → sample_freq
    fir2: Fir,
    /// Ring buffer for pass 1 (stores i32 for intermediate precision)
    buffer1: [i32; TWO_PASS_RING_SIZE * 2],
    /// Ring buffer for pass 2
    buffer2: [i32; TWO_PASS_RING_SIZE * 2],
    /// Current index in pass 1 ring buffer
    index1: usize,
    /// Current index in pass 2 ring buffer
    index2: usize,
    /// Sample offset for pass 1 (fixed-point)
    offset1: i32,
    /// Sample offset for pass 2 (fixed-point)
    offset2: i32,
    /// Cycles per sample for pass 1 (fixed-point)
    cycles_per_sample1: i32,
    /// Cycles per sample for pass 2 (fixed-point)
    cycles_per_sample2: i32,
    /// Output value from pass 1
    output_value1: i32,
    /// Final output value from pass 2
    output_value2: i32,
}

#[cfg(feature = "alloc")]
impl TwoPassResampler {
    /// Determine passband frequency based on output sample rate.
    /// Uses 20kHz for high rates (full audible range), 45% of rate for low rates.
    fn passband_freq(sample_freq: f64) -> f64 {
        if sample_freq > 44000.0 {
            TWO_PASS_DEFAULT_PASSBAND
        } else {
            sample_freq * 0.45
        }
    }

    /// Calculate optimal intermediate frequency using Laurent Ganier's formula.
    /// Minimizes total filter order by balancing decimation between two stages.
    fn calculate_intermediate_freq(clock_freq: f64, sample_freq: f64) -> f64 {
        let passband = Self::passband_freq(sample_freq);
        // Formula derived from minimizing sum of filter orders for two-stage resampling
        2.0 * passband
            + (2.0 * passband * clock_freq * (sample_freq - 2.0 * passband) / sample_freq).sqrt()
    }

    /// Create a new two-pass resampler.
    fn new(clock_freq: f64, sample_freq: f64) -> Self {
        let passband = Self::passband_freq(sample_freq);
        let intermediate_freq = Self::calculate_intermediate_freq(clock_freq, sample_freq);

        let fir1 = Self::init_fir_pass(clock_freq, intermediate_freq, passband);
        let fir2 = Self::init_fir_pass(intermediate_freq, sample_freq, passband);

        let cycles_per_sample1 =
            (clock_freq / intermediate_freq * TWO_PASS_FIXP_SCALE as f64) as i32;
        let cycles_per_sample2 =
            (intermediate_freq / sample_freq * TWO_PASS_FIXP_SCALE as f64) as i32;

        TwoPassResampler {
            fir1,
            fir2,
            buffer1: [0; TWO_PASS_RING_SIZE * 2],
            buffer2: [0; TWO_PASS_RING_SIZE * 2],
            index1: 0,
            index2: 0,
            offset1: 0,
            offset2: 0,
            cycles_per_sample1,
            cycles_per_sample2,
            output_value1: 0,
            output_value2: 0,
        }
    }

    /// Initialize FIR filter for one resampling pass.
    fn init_fir_pass(input_freq: f64, output_freq: f64, passband_freq: f64) -> Fir {
        let cycles_per_sample = input_freq / output_freq;
        let samples_per_cycle = output_freq / input_freq;

        // 16-bit target requires -96dB stopband attenuation
        let attenuation_db = -20.0_f64 * (1.0 / (1_i32 << 16) as f64).log10();

        // Transition band width (doubled since filter transitions at Nyquist)
        let transition_width =
            (1.0 - 2.0 * passband_freq / output_freq) * core::f64::consts::PI * 2.0;

        // Kaiser window parameters from MATLAB kaiserord function
        let beta = 0.1102_f64 * (attenuation_db - 8.7);
        let i0_beta = i0(beta);

        // Filter order: even for symmetric sinc around x=0
        let mut filter_order = ((attenuation_db - 7.95) / (2.285 * transition_width) + 0.5) as i32;
        filter_order += filter_order & 1;

        // Filter length: odd for symmetric sinc (order + 1)
        let mut filter_len = (filter_order as f64 * cycles_per_sample) as i32 + 1;
        filter_len |= 1;

        // Resolution bounded by interpolation error formula: err < 1.234/L^2
        let filter_res = ((1.234_f64 * (1 << 16) as f64).sqrt() * samples_per_cycle).ceil() as i32;

        let data = Self::compute_fir_coefficients(
            filter_len,
            filter_res,
            samples_per_cycle,
            beta,
            i0_beta,
        );

        Fir {
            data,
            n: filter_len,
            res: filter_res,
        }
    }

    /// Compute Kaiser-windowed sinc FIR coefficients.
    fn compute_fir_coefficients(
        filter_len: i32,
        filter_res: i32,
        samples_per_cycle: f64,
        beta: f64,
        i0_beta: f64,
    ) -> Vec<i16> {
        let mut data = alloc::vec![0i16; (filter_len * filter_res) as usize];
        let half_len = filter_len / 2;
        let half_len_f = half_len as f64;
        // Scale factor: 32768 for i16 range, samples_per_cycle for gain normalization
        let scale = 32768.0 * samples_per_cycle;

        for phase_idx in 0..filter_res {
            let phase_offset = phase_idx as f64 / filter_res as f64 + half_len_f;

            for tap_idx in 0..filter_len {
                let tap_offset = tap_idx as f64 - phase_offset;
                let normalized_pos = tap_offset / half_len_f;

                let kaiser_weight = Self::kaiser_window(normalized_pos, beta, i0_beta);
                let sinc_value = Self::sinc(tap_offset * samples_per_cycle * core::f64::consts::PI);

                data[(phase_idx * filter_len + tap_idx) as usize] =
                    (scale * sinc_value * kaiser_weight) as i16;
            }
        }
        data
    }

    /// Kaiser window function for the given normalized position [-1, 1].
    #[inline]
    fn kaiser_window(normalized_pos: f64, beta: f64, i0_beta: f64) -> f64 {
        if normalized_pos.abs() < 1.0 {
            i0(beta * sqrt_compat(1.0 - normalized_pos * normalized_pos)) / i0_beta
        } else {
            0.0
        }
    }

    /// Normalized sinc function: sin(x)/x, with sinc(0) = 1.
    #[inline]
    fn sinc(x: f64) -> f64 {
        if x.abs() >= 1e-8 {
            x.sin() / x
        } else {
            1.0
        }
    }

    /// Store sample in ring buffer with duplication for wraparound optimization.
    #[inline]
    fn store_in_ring(buffer: &mut [i32], index: &mut usize, sample: i32) {
        buffer[*index] = sample;
        buffer[*index + TWO_PASS_RING_SIZE] = sample;
        *index = (*index + 1) & TWO_PASS_RING_MASK;
    }

    /// Check if decimation produces output, update offset state.
    /// Returns (ready, offset_for_convolution) where ready indicates output available.
    #[inline]
    fn check_decimation(offset: &mut i32, cycles_per_sample: i32) -> Option<i32> {
        let convolution_offset = if *offset < TWO_PASS_FIXP_SCALE {
            let result = *offset;
            *offset += cycles_per_sample;
            Some(result)
        } else {
            None
        };
        *offset -= TWO_PASS_FIXP_SCALE;
        convolution_offset
    }

    /// Process sample through both passes, return true if final output ready.
    #[inline]
    fn input(&mut self, sample: i32) -> bool {
        // Pass 1: clock_freq → intermediate_freq
        Self::store_in_ring(&mut self.buffer1, &mut self.index1, sample);

        if let Some(offset) = Self::check_decimation(&mut self.offset1, self.cycles_per_sample1) {
            self.output_value1 =
                self.convolve_with_fir(&self.fir1, &self.buffer1, self.index1, offset);

            // Pass 2: intermediate_freq → sample_freq (only when pass 1 outputs)
            Self::store_in_ring(&mut self.buffer2, &mut self.index2, self.output_value1);

            if let Some(offset2) =
                Self::check_decimation(&mut self.offset2, self.cycles_per_sample2)
            {
                self.output_value2 =
                    self.convolve_with_fir(&self.fir2, &self.buffer2, self.index2, offset2);
                return true;
            }
        }
        false
    }

    /// Get the final output value.
    #[inline]
    fn output(&self) -> i32 {
        self.output_value2
    }

    /// FIR convolution with linear interpolation between adjacent filter tables.
    /// Interpolation improves frequency response without requiring huge FIR tables.
    fn convolve_with_fir(&self, fir: &Fir, buffer: &[i32], index: usize, subcycle: i32) -> i32 {
        // Select two adjacent FIR tables for interpolation
        let mut table_idx = (subcycle * fir.res) >> TWO_PASS_FIXP_SHIFT;
        let interp_frac = (subcycle * fir.res) & TWO_PASS_FIXP_MASK;

        // Ring buffer index for most recent samples (duplicated region avoids wrap check)
        let mut sample_idx = index + TWO_PASS_RING_SIZE - fir.n as usize;

        let v1 = Self::dot_product_i32(
            &buffer[sample_idx..sample_idx + fir.n as usize],
            &fir.data[(table_idx * fir.n) as usize..],
            fir.n as usize,
        );

        // Next table; wrap to first table requires shifting sample window
        table_idx += 1;
        if table_idx == fir.res {
            table_idx = 0;
            sample_idx += 1;
        }

        let v2 = Self::dot_product_i32(
            &buffer[sample_idx..sample_idx + fir.n as usize],
            &fir.data[(table_idx * fir.n) as usize..],
            fir.n as usize,
        );

        // Linear interpolation: v1 + frac * (v2 - v1)
        v1 + ((interp_frac * (v2 - v1)) >> TWO_PASS_FIXP_SHIFT)
    }

    /// Dot product of i32 samples with i16 coefficients.
    ///
    /// Note: Uses i32 samples (vs i16 in single-pass) to preserve precision between
    /// passes. This limits SIMD efficiency since i32×i16 requires widening, but the
    /// two-pass efficiency gain comes from shorter filters (~60+60 vs ~125 taps),
    /// not vectorization.
    #[inline]
    fn dot_product_i32(samples: &[i32], coeffs: &[i16], len: usize) -> i32 {
        let mut acc: i64 = 0;
        for i in 0..len {
            acc += samples[i] as i64 * coeffs[i] as i64;
        }
        // Rounding shift: add half the divisor before shifting
        ((acc + (1 << 14)) >> 15) as i32
    }

    /// Reset resampler state.
    fn reset(&mut self) {
        self.buffer1 = [0; TWO_PASS_RING_SIZE * 2];
        self.buffer2 = [0; TWO_PASS_RING_SIZE * 2];
        self.index1 = 0;
        self.index2 = 0;
        self.offset1 = 0;
        self.offset2 = 0;
        self.output_value1 = 0;
        self.output_value2 = 0;
    }
}

/// Compute the 0th order modified Bessel function of the first kind.
fn i0(x: f64) -> f64 {
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
fn sqrt_compat(value: f64) -> f64 {
    value.sqrt()
}

#[cfg(not(feature = "std"))]
fn sqrt_compat(value: f64) -> f64 {
    libm::sqrt(value)
}

#[derive(Clone)]
pub struct Sampler {
    // Dependencies
    pub synth: Synth,
    // Configuration
    cycles_per_sample: u32,
    #[cfg(feature = "alloc")]
    fir: Fir,
    #[cfg(feature = "alloc")]
    two_pass: Option<TwoPassResampler>,
    sampling_method: SamplingMethod,
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    use_avx2: bool,
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    use_sse2: bool,
    // Runtime State
    buffer: [i16; RING_SIZE * 2],
    index: usize,
    offset: i32,
    prev_sample: i16,
}

impl Sampler {
    pub fn new(synth: Synth) -> Self {
        Sampler {
            synth,
            cycles_per_sample: 0,
            #[cfg(feature = "alloc")]
            fir: Fir {
                data: Vec::new(),
                n: 0,
                res: 0,
            },
            #[cfg(feature = "alloc")]
            two_pass: None,
            sampling_method: SamplingMethod::Fast,
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            use_avx2: alloc::is_x86_feature_detected!("avx2"),
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            use_sse2: alloc::is_x86_feature_detected!("sse2"),
            buffer: [0; RING_SIZE * 2],
            index: 0,
            offset: 0,
            prev_sample: 0,
        }
    }

    /// Set sampling parameters.
    ///
    /// # Errors
    /// Returns `SamplingError::ZeroClockFreq` if `clock_freq` is zero.
    /// Returns `SamplingError::ZeroSampleFreq` if `sample_freq` is zero.
    pub fn set_parameters(
        &mut self,
        method: SamplingMethod,
        clock_freq: u32,
        sample_freq: u32,
    ) -> Result<(), SamplingError> {
        if clock_freq == 0 {
            return Err(SamplingError::ZeroClockFreq);
        }
        if sample_freq == 0 {
            return Err(SamplingError::ZeroSampleFreq);
        }
        self.cycles_per_sample =
            (clock_freq as f64 / sample_freq as f64 * (1 << FIXP_SHIFT) as f64 + 0.5) as u32;
        self.sampling_method = method;

        #[cfg(feature = "alloc")]
        if self.sampling_method == SamplingMethod::Resample
            || self.sampling_method == SamplingMethod::ResampleFast
        {
            self.init_fir(clock_freq as f64, sample_freq as f64, -1.0, 0.97);
            self.two_pass = None;
        }
        #[cfg(feature = "alloc")]
        if self.sampling_method == SamplingMethod::ResampleTwoPass {
            self.two_pass = Some(TwoPassResampler::new(clock_freq as f64, sample_freq as f64));
        }
        // Clear state
        for j in 0..RING_SIZE * 2 {
            self.buffer[j] = 0;
        }
        self.index = 0;
        self.offset = 0;
        self.prev_sample = 0;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.synth.reset();
        self.index = 0;
        self.offset = 0;
        self.prev_sample = 0;
        #[cfg(feature = "alloc")]
        if let Some(ref mut two_pass) = self.two_pass {
            two_pass.reset();
        }
    }

    #[inline]
    pub fn clock(&mut self, delta: u32, buffer: &mut [i16], interleave: usize) -> (usize, u32) {
        match self.sampling_method {
            SamplingMethod::Fast => self.clock_fast(delta, buffer, interleave),
            SamplingMethod::Interpolate => self.clock_interpolate(delta, buffer, interleave),
            #[cfg(feature = "alloc")]
            SamplingMethod::Resample => self.clock_resample_interpolate(delta, buffer, interleave),
            #[cfg(feature = "alloc")]
            SamplingMethod::ResampleFast => self.clock_resample_fast(delta, buffer, interleave),
            #[cfg(feature = "alloc")]
            SamplingMethod::ResampleTwoPass => {
                self.clock_resample_two_pass(delta, buffer, interleave)
            }
        }
    }

    /// SID clocking with audio sampling - delta clocking picking nearest sample.
    #[inline]
    fn clock_fast(
        &mut self,
        mut delta: u32,
        buffer: &mut [i16],
        interleave: usize,
    ) -> (usize, u32) {
        let mut index = 0;
        loop {
            let next_sample_offset = self.get_next_sample_offset();
            let delta_sample = (next_sample_offset >> FIXP_SHIFT) as u32;
            if delta_sample > delta || index >= buffer.len() {
                break;
            }
            self.synth.clock_delta(delta_sample);
            delta -= delta_sample;
            buffer[index * interleave] = self.synth.output();
            index += 1;
            self.update_sample_offset(next_sample_offset);
        }
        if delta > 0 && index < buffer.len() {
            self.synth.clock_delta(delta);
            self.offset -= (delta as i32) << FIXP_SHIFT;
            (index, 0)
        } else {
            (index, delta)
        }
    }

    #[inline]
    fn clock_interpolate(
        &mut self,
        mut delta: u32,
        buffer: &mut [i16],
        interleave: usize,
    ) -> (usize, u32) {
        let mut index = 0;
        loop {
            let next_sample_offset = self.get_next_sample_offset();
            let delta_sample = (next_sample_offset >> FIXP_SHIFT) as u32;
            if delta_sample > delta || index >= buffer.len() {
                break;
            }
            for _i in 0..(delta_sample - 1) {
                self.prev_sample = self.synth.output();
                self.synth.clock();
            }
            delta -= delta_sample;
            let sample_now = self.synth.output();
            buffer[index * interleave] = self.prev_sample
                + ((self.offset * (sample_now - self.prev_sample) as i32) >> FIXP_SHIFT) as i16;
            index += 1;
            self.prev_sample = sample_now;
            self.update_sample_offset(next_sample_offset);
        }
        if delta > 0 && index < buffer.len() {
            for _i in 0..(delta - 1) {
                self.synth.clock();
            }
            self.offset -= (delta as i32) << FIXP_SHIFT;
            (index, 0)
        } else {
            (index, delta)
        }
    }

    /// SID clocking with audio sampling - cycle based with audio resampling.
    ///
    /// This is the theoretically correct (and computationally intensive) audio
    /// sample generation. The samples are generated by resampling to the specified
    /// sampling frequency. The work rate is inversely proportional to the
    /// percentage of the bandwidth allocated to the filter transition band.
    ///
    /// This implementation is based on the paper "A Flexible Sampling-Rate
    /// Conversion Method", by J. O. Smith and P. Gosset, or rather on the
    /// expanded tutorial on the "Digital Audio Resampling Home Page":
    /// http://www-ccrma.stanford.edu/~jos/resample/
    ///
    /// By building shifted FIR tables with samples according to the
    /// sampling frequency, this implementation dramatically reduces the
    /// computational effort in the filter convolutions, without any loss
    /// of accuracy. The filter convolutions are also vectorizable on
    /// current hardware.
    ///
    /// Further possible optimizations are:
    /// * An equiripple filter design could yield a lower filter order, see
    ///   http://www.mwrf.com/Articles/ArticleID/7229/7229.html
    /// * The Convolution Theorem could be used to bring the complexity of
    ///   convolution down from O(n*n) to O(n*log(n)) using the Fast Fourier
    ///   Transform, see http://en.wikipedia.org/wiki/Convolution_theorem
    /// * Simply resampling in two steps can also yield computational
    ///   savings, since the transition band will be wider in the first step
    ///   and the required filter order is thus lower in this step.
    ///   Laurent Ganier has found the optimal intermediate sampling frequency
    ///   to be (via derivation of sum of two steps):
    ///   2 * pass_freq + sqrt [ 2 * pass_freq * orig_sample_freq
    ///   * (dest_sample_freq - 2 * pass_freq) / dest_sample_freq ]
    ///
    /// NB! the result of right shifting negative numbers is really
    /// implementation dependent in the C++ standard.
    #[cfg(feature = "alloc")]
    #[inline]
    fn clock_resample_interpolate(
        &mut self,
        mut delta: u32,
        buffer: &mut [i16],
        interleave: usize,
    ) -> (usize, u32) {
        let mut index = 0;
        loop {
            let next_sample_offset = self.get_next_sample_offset2();
            let delta_sample = (next_sample_offset >> FIXP_SHIFT) as u32;
            if delta_sample > delta || index >= buffer.len() {
                break;
            }

            for _i in 0..delta_sample {
                self.synth.clock();
                let output = self.synth.output();
                self.buffer[self.index] = output;
                self.buffer[self.index + RING_SIZE] = output;
                self.index += 1;
                self.index &= RING_MASK;
            }
            delta -= delta_sample;
            self.update_sample_offset2(next_sample_offset);

            let fir_offset_1 = (self.offset * self.fir.res) >> FIXP_SHIFT;
            let fir_offset_rmd = (self.offset * self.fir.res) & FIXP_MASK;
            let fir_start_1 = (fir_offset_1 * self.fir.n) as usize;
            let fir_end_1 = fir_start_1 + self.fir.n as usize;
            let sample_start_1 = (self.index as i32 - self.fir.n + RING_SIZE as i32) as usize;
            let sample_end_1 = sample_start_1 + self.fir.n as usize;

            // Convolution with filter impulse response.
            let v1 = self.compute_convolution_fir(
                &self.buffer[sample_start_1..sample_end_1],
                &self.fir.data[fir_start_1..fir_end_1],
            );

            // Use next FIR table, wrap around to first FIR table using
            // previous sample.
            let mut fir_offset_2 = fir_offset_1 + 1;
            let mut sample_start_2 = sample_start_1;
            if fir_offset_2 == self.fir.res {
                fir_offset_2 = 0;
                sample_start_2 -= 1;
            }
            let fir_start_2 = (fir_offset_2 * self.fir.n) as usize;
            let fir_end_2 = fir_start_2 + self.fir.n as usize;
            let sample_end_2 = sample_start_2 + self.fir.n as usize;

            let v2 = self.compute_convolution_fir(
                &self.buffer[sample_start_2..sample_end_2],
                &self.fir.data[fir_start_2..fir_end_2],
            );

            // Linear interpolation.
            // fir_offset_rmd is equal for all samples, it can thus be factorized out:
            // sum(v1 + rmd*(v2 - v1)) = sum(v1) + rmd*(sum(v2) - sum(v1))
            let mut v = v1 + ((fir_offset_rmd * (v2 - v1)) >> FIXP_SHIFT);
            v >>= FIR_SHIFT;

            // Soft clip for smooth saturation near 16-bit boundaries
            buffer[index * interleave] = soft_clip(v);
            index += 1;
        }
        if delta > 0 && index < buffer.len() {
            for _i in 0..delta {
                self.synth.clock();
                let output = self.synth.output();
                self.buffer[self.index] = output;
                self.buffer[self.index + RING_SIZE] = output;
                self.index += 1;
                self.index &= RING_MASK;
            }
            self.offset -= (delta as i32) << FIXP_SHIFT;
            (index, 0)
        } else {
            (index, delta)
        }
    }

    /// SID clocking with audio sampling - cycle based with audio resampling.
    #[cfg(feature = "alloc")]
    #[inline]
    fn clock_resample_fast(
        &mut self,
        mut delta: u32,
        buffer: &mut [i16],
        interleave: usize,
    ) -> (usize, u32) {
        let mut index = 0;
        loop {
            let next_sample_offset = self.get_next_sample_offset2();
            let delta_sample = (next_sample_offset >> FIXP_SHIFT) as u32;
            if delta_sample > delta || index >= buffer.len() {
                break;
            }

            for _i in 0..delta_sample {
                self.synth.clock();
                let output = self.synth.output();
                self.buffer[self.index] = output;
                self.buffer[self.index + RING_SIZE] = output;
                self.index += 1;
                self.index &= RING_MASK;
            }
            delta -= delta_sample;
            self.update_sample_offset2(next_sample_offset);

            let fir_offset = (self.offset * self.fir.res) >> FIXP_SHIFT;
            let fir_start = (fir_offset * self.fir.n) as usize;
            let fir_end = fir_start + self.fir.n as usize;
            let sample_start = (self.index as i32 - self.fir.n + RING_SIZE as i32) as usize;
            let sample_end = sample_start + self.fir.n as usize;

            // Convolution with filter impulse response.
            let mut v = self.compute_convolution_fir(
                &self.buffer[sample_start..sample_end],
                &self.fir.data[fir_start..fir_end],
            );
            v >>= FIR_SHIFT;

            // Soft clip for smooth saturation near 16-bit boundaries
            buffer[index * interleave] = soft_clip(v);
            index += 1;
        }
        if delta > 0 && index < buffer.len() {
            for _i in 0..delta {
                self.synth.clock();
                let output = self.synth.output();
                self.buffer[self.index] = output;
                self.buffer[self.index + RING_SIZE] = output;
                self.index += 1;
                self.index &= RING_MASK;
            }
            self.offset -= (delta as i32) << FIXP_SHIFT;
            (index, 0)
        } else {
            (index, delta)
        }
    }

    /// SID clocking with two-pass audio resampling.
    ///
    /// Chains two FIR filters through an intermediate frequency for improved
    /// computational efficiency. The intermediate frequency (~120 kHz) is chosen
    /// to minimize total filter order using Laurent Ganier's formula.
    #[cfg(feature = "alloc")]
    #[inline]
    fn clock_resample_two_pass(
        &mut self,
        delta: u32,
        buffer: &mut [i16],
        interleave: usize,
    ) -> (usize, u32) {
        let two_pass = self
            .two_pass
            .as_mut()
            .expect("TwoPassResampler not initialized");

        let mut index = 0;

        // Clock SID and feed samples through two-pass resampler
        for _ in 0..delta {
            if index >= buffer.len() {
                break;
            }

            self.synth.clock();
            let sample = self.synth.output() as i32;

            if two_pass.input(sample) {
                buffer[index * interleave] = soft_clip(two_pass.output());
                index += 1;
            }
        }

        // Calculate remaining delta
        let consumed = if index < buffer.len() { delta } else { 0 };
        (index, delta - consumed)
    }

    /// Dispatches to AVX2 intrinsics if available, wide_256 for SSE2, otherwise fallback.
    /// LLVM auto-vectorizes fallback well on NEON.
    #[inline]
    pub fn compute_convolution_fir(&self, sample: &[i16], fir: &[i16]) -> i32 {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if self.use_avx2 {
                return unsafe { self.compute_convolution_fir_avx2(sample, fir) };
            }
            if self.use_sse2 {
                return self.compute_convolution_fir_wide_256(sample, fir);
            }
        }
        self.compute_convolution_fir_fallback(sample, fir)
    }

    /// LLVM auto-vectorizes this well on SSE/NEON.
    #[inline]
    pub fn compute_convolution_fir_fallback(&self, sample: &[i16], fir: &[i16]) -> i32 {
        let len = sample.len().min(fir.len());
        sample[..len]
            .iter()
            .zip(&fir[..len])
            .fold(0, |sum, (&s, &f)| sum + (s as i32 * f as i32))
    }

    /// Uses wide crate for portable SIMD; emits vpmaddwd on AVX2.
    #[inline]
    pub fn compute_convolution_fir_wide_256(&self, sample: &[i16], fir: &[i16]) -> i32 {
        let len = sample.len().min(fir.len());
        let mut ss = &sample[..len];
        let mut fs = &fir[..len];

        // 4 accumulators hide instruction latency
        let mut v1 = i32x8::ZERO;
        let mut v2 = i32x8::ZERO;
        let mut v3 = i32x8::ZERO;
        let mut v4 = i32x8::ZERO;

        while ss.len() >= 64 {
            let sv1 = i16x16::from(&ss[0..16]);
            let sv2 = i16x16::from(&ss[16..32]);
            let sv3 = i16x16::from(&ss[32..48]);
            let sv4 = i16x16::from(&ss[48..64]);
            let fv1 = i16x16::from(&fs[0..16]);
            let fv2 = i16x16::from(&fs[16..32]);
            let fv3 = i16x16::from(&fs[32..48]);
            let fv4 = i16x16::from(&fs[48..64]);

            v1 += sv1.dot(fv1);
            v2 += sv2.dot(fv2);
            v3 += sv3.dot(fv3);
            v4 += sv4.dot(fv4);

            ss = &ss[64..];
            fs = &fs[64..];
        }

        let combined = v1 + v2 + v3 + v4;
        let mut v = combined.reduce_add();

        for i in 0..ss.len() {
            v += ss[i] as i32 * fs[i] as i32;
        }
        v
    }

    #[target_feature(enable = "avx2")]
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    pub unsafe fn compute_convolution_fir_avx2(&self, sample: &[i16], fir: &[i16]) -> i32 {
        #[cfg(target_arch = "x86")]
        use alloc::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use alloc::arch::x86_64::*;

        // Convolution with filter impulse response.
        let len = alloc::cmp::min(sample.len(), fir.len());
        let mut fs = &fir[..len];
        let mut ss = &sample[..len];
        // 4 accumulators hide instruction latency
        let mut v1 = _mm256_set1_epi32(0);
        let mut v2 = _mm256_set1_epi32(0);
        let mut v3 = _mm256_set1_epi32(0);
        let mut v4 = _mm256_set1_epi32(0);
        while fs.len() >= 64 {
            let sv1 = _mm256_loadu_si256(ss.as_ptr() as *const _);
            let sv2 = _mm256_loadu_si256((&ss[16..]).as_ptr() as *const _);
            let sv3 = _mm256_loadu_si256((&ss[32..]).as_ptr() as *const _);
            let sv4 = _mm256_loadu_si256((&ss[48..]).as_ptr() as *const _);
            let fv1 = _mm256_loadu_si256(fs.as_ptr() as *const _);
            let fv2 = _mm256_loadu_si256((&fs[16..]).as_ptr() as *const _);
            let fv3 = _mm256_loadu_si256((&fs[32..]).as_ptr() as *const _);
            let fv4 = _mm256_loadu_si256((&fs[48..]).as_ptr() as *const _);
            let prod1 = _mm256_madd_epi16(sv1, fv1);
            let prod2 = _mm256_madd_epi16(sv2, fv2);
            let prod3 = _mm256_madd_epi16(sv3, fv3);
            let prod4 = _mm256_madd_epi16(sv4, fv4);
            v1 = _mm256_add_epi32(v1, prod1);
            v2 = _mm256_add_epi32(v2, prod2);
            v3 = _mm256_add_epi32(v3, prod3);
            v4 = _mm256_add_epi32(v4, prod4);
            fs = &fs[64..];
            ss = &ss[64..];
        }
        v1 = _mm256_add_epi32(v1, v2);
        v3 = _mm256_add_epi32(v3, v4);
        v1 = _mm256_add_epi32(v1, v3);
        let mut va = [0i32; 8];
        _mm256_storeu_si256(va[..].as_mut_ptr() as *mut _, v1);
        let mut v = va[0] + va[1] + va[2] + va[3] + va[4] + va[5] + va[6] + va[7];
        for i in 0..fs.len() {
            v += ss[i] as i32 * fs[i] as i32;
        }
        v
    }

    #[inline]
    fn get_next_sample_offset(&self) -> i32 {
        self.offset + self.cycles_per_sample as i32 + (1 << (FIXP_SHIFT - 1))
    }

    #[inline]
    fn get_next_sample_offset2(&self) -> i32 {
        self.offset + self.cycles_per_sample as i32
    }

    #[inline]
    fn update_sample_offset(&mut self, next_sample_offset: i32) {
        self.offset = (next_sample_offset & FIXP_MASK) - (1 << (FIXP_SHIFT - 1));
    }

    #[inline]
    fn update_sample_offset2(&mut self, next_sample_offset: i32) {
        self.offset = next_sample_offset & FIXP_MASK;
    }

    #[cfg(feature = "alloc")]
    fn init_fir(
        &mut self,
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
        let atten = -20.0f64 * (1.0 / (1i32 << 16) as f64).log10();
        // A fraction of the bandwidth is allocated to the transition band,
        let dw = (1.0f64 - 2.0 * pass_freq / sample_freq) * pi;
        // The cutoff frequency is midway through the transition band.
        let wc = (2.0f64 * pass_freq / sample_freq + 1.0) * pi / 2.0;

        // For calculation of beta and N see the reference for the kaiserord
        // function in the MATLAB Signal Processing Toolbox:
        // http://www.mathworks.com/access/helpdesk/help/toolbox/signal/kaiserord.html
        let beta = 0.1102f64 * (atten - 8.7);
        let io_beta = i0(beta);

        // The filter order will maximally be 124 with the current constraints.
        // N >= (96.33 - 7.95)/(2.285*0.1*pi) -> N >= 123
        // The filter order is equal to the number of zero crossings, i.e.
        // it should be an even number (sinc is symmetric about x = 0).
        let mut n_cap = ((atten - 7.95) / (2.285 * dw) + 0.5) as i32;
        n_cap += n_cap & 1;

        // The filter length is equal to the filter order + 1.
        // The filter length must be an odd number (sinc is symmetric about x = 0).
        self.fir.n = (n_cap as f64 * cycles_per_sample) as i32 + 1;
        self.fir.n |= 1;

        // We clamp the filter table resolution to 2^n, making the fixpoint
        // sample_offset a whole multiple of the filter table resolution.
        let res = if self.sampling_method == SamplingMethod::Resample {
            FIR_RES_INTERPOLATE
        } else {
            FIR_RES_FAST
        };
        let n = ((res as f64 / cycles_per_sample).ln() / (2.0f64).ln()).ceil() as i32;
        self.fir.res = 1 << n;

        self.fir.data.clear();
        self.fir
            .data
            .resize((self.fir.n * self.fir.res) as usize, 0);

        // Calculate fir_RES FIR tables for linear interpolation.
        for i in 0..self.fir.res {
            let fir_offset = i * self.fir.n + self.fir.n / 2;
            let j_offset = i as f64 / self.fir.res as f64;
            // Calculate FIR table. This is the sinc function, weighted by the
            // Kaiser window.
            let fir_n_div2 = self.fir.n / 2;
            for j in -fir_n_div2..=fir_n_div2 {
                let jx = j as f64 - j_offset;
                let wt = wc * jx / cycles_per_sample;
                let temp = jx / fir_n_div2 as f64;
                let kaiser = if temp.abs() <= 1.0 {
                    i0(beta * sqrt_compat(1.0 - temp * temp)) / io_beta
                } else {
                    0f64
                };
                let sincwt = if wt.abs() >= 1e-6 { wt.sin() / wt } else { 1.0 };
                let val = (1i32 << FIR_SHIFT) as f64 * filter_scale * samples_per_cycle * wc / pi
                    * sincwt
                    * kaiser;
                self.fir.data[(fir_offset + j) as usize] = (val + 0.5) as i16;
            }
        }
    }
}
