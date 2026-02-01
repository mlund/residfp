// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

//! Audio sampling and resampling for SID output.
//!
//! Provides multiple sampling methods from fast decimation to high-quality
//! Kaiser-windowed sinc resampling.

// Allow cast_lossless: intentional i16->i32 casts for audio sample processing
#![allow(clippy::cast_lossless)]
// Allow cast_ptr_alignment: SIMD pointer casts are aligned by construction
#![allow(clippy::cast_ptr_alignment)]

#[cfg(feature = "alloc")]
mod fir;
mod soft_clip;
#[cfg(feature = "alloc")]
mod two_pass;

pub use soft_clip::soft_clip;

use crate::synth::Synth;
use crate::SamplingError;

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

/// Audio sampling/resampling method.
///
/// Controls how SID output is converted to the target sample rate.
/// Methods requiring heap allocation are gated behind the `alloc` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SamplingMethod {
    /// Simple decimation - fastest but lowest quality.
    #[default]
    Fast,
    /// Linear interpolation between samples.
    Interpolate,
    /// High-quality Kaiser-windowed sinc resampling (requires `alloc`).
    #[cfg(feature = "alloc")]
    Resample,
    /// Faster sinc resampling with larger lookup tables (requires `alloc`).
    #[cfg(feature = "alloc")]
    ResampleFast,
    /// Two-pass sinc resampling for efficiency at high ratios (requires `alloc`).
    #[cfg(feature = "alloc")]
    ResampleTwoPass,
}

#[derive(Clone)]
/// Audio sampler wrapping the SID synthesizer and resamplers.
pub struct Sampler {
    // Dependencies
    /// Underlying SID synthesizer.
    pub synth: Synth,
    // Configuration
    cycles_per_sample: u32,
    #[cfg(feature = "alloc")]
    fir: fir::Fir,
    #[cfg(feature = "alloc")]
    two_pass: Option<two_pass::TwoPassResampler>,
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
    /// Construct a sampler around a SID synthesizer.
    pub fn new(synth: Synth) -> Self {
        Self {
            synth,
            cycles_per_sample: 0,
            #[cfg(feature = "alloc")]
            fir: fir::Fir::default(),
            #[cfg(feature = "alloc")]
            two_pass: None,
            sampling_method: SamplingMethod::Fast,
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            use_avx2: std::is_x86_feature_detected!("avx2"),
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            use_sse2: std::is_x86_feature_detected!("sse2"),
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
            fir::init_fir(
                &mut self.fir,
                self.sampling_method,
                clock_freq as f64,
                sample_freq as f64,
                -1.0,
                0.97,
            );
            self.two_pass = None;
        }
        #[cfg(feature = "alloc")]
        if self.sampling_method == SamplingMethod::ResampleTwoPass {
            self.two_pass = Some(two_pass::TwoPassResampler::new(
                clock_freq as f64,
                sample_freq as f64,
            ));
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

    /// Reset sampler and underlying synth/filter state.
    /// Reset sampler and underlying synth/filter state.
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
    /// Clock the sampler for `delta` cycles, writing interleaved audio samples.
    /// Clock the sampler for `delta` SID cycles, writing interleaved audio samples.
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
    /// Nearest-neighbor (decimation) sampling.
    /// Nearest-neighbor (decimation) sampling.
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
    /// Linear interpolation sampling.
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
    #[cfg(feature = "alloc")]
    #[inline]
    /// High-quality sinc resampling (Kaiser window).
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
    /// Faster sinc resampling using larger precomputed tables.
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
    #[cfg(feature = "alloc")]
    #[inline]
    /// Two-pass sinc resampling for large ratio changes.
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
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let len = sample.len().min(fir.len());
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
    const fn get_next_sample_offset(&self) -> i32 {
        self.offset + self.cycles_per_sample as i32 + (1 << (FIXP_SHIFT - 1))
    }

    #[inline]
    const fn get_next_sample_offset2(&self) -> i32 {
        self.offset + self.cycles_per_sample as i32
    }

    #[inline]
    const fn update_sample_offset(&mut self, next_sample_offset: i32) {
        self.offset = (next_sample_offset & FIXP_MASK) - (1 << (FIXP_SHIFT - 1));
    }

    #[inline]
    const fn update_sample_offset2(&mut self, next_sample_offset: i32) {
        self.offset = next_sample_offset & FIXP_MASK;
    }
}
