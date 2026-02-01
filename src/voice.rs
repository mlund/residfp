// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

#![allow(clippy::cast_lossless)]

use super::envelope::EnvelopeGenerator;
use super::wave::{Syncable, WaveformGenerator};
use super::ChipModel;

/// The waveform output range is 0x000 to 0xfff, so the "zero"
/// level should ideally have been 0x800. In the measured chip, the
/// waveform output "zero" level was found to be 0x380 (i.e. $d41b
/// = 0x38) at 5.94V.
const WAVE_ZERO: i32 = 0x0380;

/// The envelope multiplying D/A converter introduces another DC
/// offset. This is isolated by the following measurements:
///
/// * The "zero" output level of the mixer at full volume is 5.44V.
/// * Routing one voice to the mixer at full volume yields
///   6.75V at maximum voice output (wave = 0xfff, sustain = 0xf)
///   5.94V at "zero" voice output  (wave = any,   sustain = 0x0)
///   5.70V at minimum voice output (wave = 0x000, sustain = 0xf)
/// * The DC offset of one voice is (5.94V - 5.44V) = 0.50V
/// * The dynamic range of one voice is |6.75V - 5.70V| = 1.05V
/// * The DC offset is thus 0.50V/1.05V ~ 1/2 of the dynamic range.
///
/// Note that by removing the DC offset, we get the following ranges for
/// one voice:
///     y > 0: (6.75V - 5.44V) - 0.50V =  0.81V
///     y < 0: (5.70V - 5.44V) - 0.50V = -0.24V
/// The scaling of the voice amplitude is not symmetric about y = 0;
/// this follows from the DC level in the waveform output.
const VOICE_DC: i32 = 0x800 * 0xff;

/// A single SID voice combining waveform and envelope generators.
#[derive(Clone)]
pub struct Voice {
    // Configuration
    wave_zero: i32,
    voice_dc: i32,
    // Generators
    pub(crate) envelope: EnvelopeGenerator,
    pub(crate) wave: WaveformGenerator,
}

impl Voice {
    /// Create a voice for the given chip model.
    pub fn new(chip_model: ChipModel) -> Self {
        match chip_model {
            ChipModel::Mos6581 => Voice {
                wave_zero: WAVE_ZERO,
                voice_dc: VOICE_DC,
                envelope: EnvelopeGenerator::default(),
                wave: WaveformGenerator::new(chip_model),
            },
            ChipModel::Mos8580 => Voice {
                // No DC offsets in the MOS8580.
                wave_zero: 0x800,
                voice_dc: 0,
                envelope: EnvelopeGenerator::default(),
                wave: WaveformGenerator::new(chip_model),
            },
        }
    }

    /// Update envelope and waveform control registers.
    pub fn set_control(&mut self, value: u8) {
        self.envelope.set_control(value);
        self.wave.set_control(value);
    }

    /// Amplitude modulated waveform output using DAC nonlinearity model.
    /// The waveform and envelope values are looked up in DAC tables to
    /// emulate R-2R ladder imperfections (6581) or linear response (8580).
    #[inline]
    pub fn output_dac(
        &self,
        sync_source: Option<&WaveformGenerator>,
        wav_dac: &[f32],
        env_dac: &[f32],
    ) -> i32 {
        let wav = self.wave.output(sync_source) as usize;
        let env = self.envelope.output() as usize;
        // DAC tables are normalized to [0, 1], scale to voice output range
        // Range: approximately [-2048*255, 2047*255] centered around voice_dc
        let wav_analog = wav_dac[wav] - wav_dac[self.wave_zero as usize];
        let env_analog = env_dac[env];
        // Scale to 20-bit range and add DC offset
        let output = wav_analog * env_analog * (4096.0 * 256.0);
        output as i32 + self.voice_dc
    }

    /// Amplitude modulated 20-bit waveform output (legacy linear model).
    /// Range [-2048*255, 2047*255].
    #[inline]
    pub fn output(&self, sync_source: Option<&WaveformGenerator>) -> i32 {
        // Multiply oscillator output with envelope output.
        (self.wave.output(sync_source) as i32 - self.wave_zero) * self.envelope.output() as i32
            + self.voice_dc
    }

    /// Reset waveform and envelope state.
    pub fn reset(&mut self) {
        self.envelope.reset();
        self.wave.reset();
    }
}

impl Syncable<&'_ Voice> {
    /// Output mixed waveform*envelope for the main voice with sync applied.
    pub fn output(&self) -> i32 {
        self.main.output(Some(&self.sync_source.wave))
    }

    /// Output using DAC lookup tables for nonlinearity modeling.
    pub fn output_dac(&self, wav_dac: &[f32], env_dac: &[f32]) -> i32 {
        self.main
            .output_dac(Some(&self.sync_source.wave), wav_dac, env_dac)
    }
}

impl<'a> Syncable<&'a Voice> {
    /// Access waveform generators for sync relationships (immutable).
    pub fn wave(self) -> Syncable<&'a WaveformGenerator> {
        Syncable {
            main: &self.main.wave,
            sync_dest: &self.sync_dest.wave,
            sync_source: &self.sync_source.wave,
        }
    }
}

impl<'a> Syncable<&'a mut Voice> {
    /// Access waveform generators for sync relationships (mutable).
    pub fn wave(self) -> Syncable<&'a mut WaveformGenerator> {
        Syncable {
            main: &mut self.main.wave,
            sync_dest: &mut self.sync_dest.wave,
            sync_source: &mut self.sync_source.wave,
        }
    }
}
