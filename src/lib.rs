// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

#![no_std]
#![warn(missing_docs)]
//! Floating-point SID (MOS6581/8580) emulator derived from libresidfp.
//!
//! ## Feature flags
//! - `ekv-filter`: Enables a physics-based EKV transistor model for the 6581
//!   filter. Improves accuracy (especially on darker 6581 chips) at the cost of
//!   ~400KB of lookup tables and additional CPU usage. No effect on 8580.

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;
#[cfg(all(feature = "alloc", feature = "std"))]
extern crate std as alloc;
pub mod dac;
mod data;
/// Envelope generator modeling SID ADSR behavior.
pub mod envelope;
/// External C64 audio output filter.
pub mod external_filter;
/// Internal SID multimode filter implementation.
pub mod filter;
#[cfg(feature = "ekv-filter")]
pub mod filter_ekv;
pub mod sampler;
mod sid;
pub mod spline;
/// Core SID synthesizer combining voices, filter, and routing.
pub mod synth;
/// Voice primitives (waveform + envelope).
pub mod voice;
/// Oscillator waveform generator primitives and sync helpers.
pub mod wave;

/// Configuration for constructing a [`Sid`].
#[cfg(all(feature = "alloc", feature = "std"))]
pub use self::sid::SidConfig;

/// SID chip model selection.
///
/// The MOS 6581 was the original SID chip used in early C64s, featuring
/// a distinctive filter with analog imperfections. The MOS 8580 was a
/// later revision with a cleaner, more linear filter response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ChipModel {
    /// Original SID chip (1982) with characteristic analog filter quirks.
    #[default]
    Mos6581,
    /// Revised SID chip (1987) with cleaner, more linear filter.
    Mos8580,
}

/// Clock frequency constants for common C64 configurations.
pub mod clock {
    /// PAL C64 clock frequency (~985 kHz).
    pub const PAL: u32 = 985_248;
    /// NTSC C64 clock frequency (~1.02 MHz).
    pub const NTSC: u32 = 1_022_727;
}

pub use self::sampler::SamplingMethod;
pub use self::sid::{Sid, State};

/// Error returned when sampling parameters are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingError {
    /// Clock frequency must be non-zero.
    ZeroClockFreq,
    /// Sample frequency must be non-zero.
    ZeroSampleFreq,
}
