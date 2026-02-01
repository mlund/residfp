// This file is part of resid-rs.
// Copyright (c) 2017-2019 Sebastian Jastrzebski <sebby2k@gmail.com>. All rights reserved.
// Portions (c) 2004 Dag Lem <resid@nimrod.no>
// Licensed under the GPLv3. See LICENSE file in the project root for full license text.

#![allow(clippy::cast_lossless)]

use bit_field::BitField;

const RATE_COUNTER_MASK: u16 = 0x7fff;
const RATE_COUNTER_MSB_MASK: u16 = 0x8000;

// Rate counter periods are calculated from the Envelope Rates table in
// the Programmer's Reference Guide. The rate counter period is the number of
// cycles between each increment of the envelope counter.
// The rates have been verified by sampling ENV3.
//
// The rate counter is a 16 bit register which is incremented each cycle.
// When the counter reaches a specific comparison value, the envelope counter
// is incremented (attack) or decremented (decay/release) and the
// counter is zeroed.
//
// NB! Sampling ENV3 shows that the calculated values are not exact.
// It may seem like most calculated values have been rounded (.5 is rounded
// down) and 1 has beed added to the result. A possible explanation for this
// is that the SID designers have used the calculated values directly
// as rate counter comparison values, not considering a one cycle delay to
// zero the counter. This would yield an actual period of comparison value + 1.
//
// The time of the first envelope count can not be exactly controlled, except
// possibly by resetting the chip. Because of this we cannot do cycle exact
// sampling and must devise another method to calculate the rate counter
// periods.
//
// The exact rate counter periods can be determined e.g. by counting the number
// of cycles from envelope level 1 to envelope level 129, and dividing the
// number of cycles by 128. CIA1 timer A and B in linked mode can perform
// the cycle count. This is the method used to find the rates below.
//
// To avoid the ADSR delay bug, sampling of ENV3 should be done using
// sustain = release = 0. This ensures that the attack state will not lower
// the current rate counter period.
//
// The ENV3 sampling code below yields a maximum timing error of 14 cycles.
//     lda #$01
// l1: cmp $d41c
//     bne l1
//     ...
//     lda #$ff
// l2: cmp $d41c
//     bne l2
//
// This yields a maximum error for the calculated rate period of 14/128 cycles.
// The described method is thus sufficient for exact calculation of the rate
// periods.
//
const RATE_COUNTER_PERIOD: [u16; 16] = [
    9,     // 2ms*1.0MHz/256 = 7.81
    32,    // 8ms*1.0MHz/256 = 31.25
    63,    // 16ms*1.0MHz/256 = 62.50
    95,    // 24ms*1.0MHz/256 = 93.75
    149,   // 38ms*1.0MHz/256 = 148.44
    220,   // 56ms*1.0MHz/256 = 218.75
    267,   // 68ms*1.0MHz/256 = 265.63
    313,   // 80ms*1.0MHz/256 = 312.50
    392,   // 100ms*1.0MHz/256 = 390.63
    977,   // 250ms*1.0MHz/256 = 976.56
    1954,  // 500ms*1.0MHz/256 = 1953.13
    3126,  // 800ms*1.0MHz/256 = 3125.00
    3907,  // 1 s*1.0MHz/256 =  3906.25
    11720, // 3 s*1.0MHz/256 = 11718.75
    19532, // 5 s*1.0MHz/256 = 19531.25
    31251, // 8 s*1.0MHz/256 = 31250.00
];

/// From the sustain levels it follows that both the low and high 4 bits of the
/// envelope counter are compared to the 4-bit sustain value.
/// This has been verified by sampling ENV3.
const SUSTAIN_LEVEL: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
];

/// Envelope generator state machine.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Attack phase ramping up toward 0xff.
    Attack,
    /// Decay toward sustain, then hold.
    DecaySustain,
    /// Release toward zero after gate off.
    Release,
}

/// SID ADSR envelope generator.
///
/// A 15 bit counter implements envelope rates by dividing the clock to the
/// envelope counter by the currently selected rate period. Another counter
/// implements exponential decay, with periods 1, 2, 4, 8, 16, 30 at envelope
/// values 255, 93, 54, 26, 14, 6 respectively.
#[derive(Clone, Copy)]
pub struct EnvelopeGenerator {
    // Configuration
    attack: u8,
    decay: u8,
    sustain: u8,
    release: u8,
    // Control
    gate: bool,
    // Runtime State
    /// Current ADSR phase.
    pub state: State,
    /// Current envelope output level (0-255).
    pub envelope_counter: u8,
    /// Exponential step counter.
    pub exponential_counter: u8,
    /// Exponential counter period.
    pub exponential_counter_period: u8,
    /// Holds zero when test/underflow.
    pub hold_zero: bool,
    /// Linear rate counter.
    pub rate_counter: u16,
    /// Linear rate counter period.
    pub rate_counter_period: u16,
}

impl Default for EnvelopeGenerator {
    fn default() -> Self {
        let mut envelope = Self {
            attack: 0,
            decay: 0,
            sustain: 0,
            release: 0,
            gate: false,
            state: State::Release,
            envelope_counter: 0,
            exponential_counter: 0,
            exponential_counter_period: 0,
            hold_zero: false,
            rate_counter: 0,
            rate_counter_period: 0,
        };
        envelope.reset();
        envelope
    }
}

impl EnvelopeGenerator {
    /// Packed attack/decay nibble register.
    pub const fn get_attack_decay(&self) -> u8 {
        self.attack << 4 | self.decay
    }

    /// Control register exposing gate bit.
    pub fn get_control(&self) -> u8 {
        let mut value = 0u8;
        value.set_bit(0, self.gate);
        value
    }

    /// Packed sustain/release nibble register.
    pub const fn get_sustain_release(&self) -> u8 {
        self.sustain << 4 | self.release
    }

    /// Write attack/decay register.
    pub const fn set_attack_decay(&mut self, value: u8) {
        self.attack = (value >> 4) & 0x0f;
        self.decay = value & 0x0f;
        match self.state {
            State::Attack => self.rate_counter_period = RATE_COUNTER_PERIOD[self.attack as usize],
            State::DecaySustain => {
                self.rate_counter_period = RATE_COUNTER_PERIOD[self.decay as usize]
            }
            _ => {}
        }
    }

    /// Write control register (gate).
    pub fn set_control(&mut self, value: u8) {
        let gate = value.get_bit(0);
        if !self.gate && gate {
            // Gate bit on: Start attack, decay, sustain.
            self.state = State::Attack;
            self.rate_counter_period = RATE_COUNTER_PERIOD[self.attack as usize];
            // Switching to attack state unlocks the zero freeze.
            self.hold_zero = false;
        } else if self.gate && !gate {
            // Gate bit off: Start release.
            self.state = State::Release;
            self.rate_counter_period = RATE_COUNTER_PERIOD[self.release as usize];
        }
        self.gate = gate;
    }

    /// Write sustain/release register.
    pub fn set_sustain_release(&mut self, value: u8) {
        self.sustain = (value >> 4) & 0x0f;
        self.release = value & 0x0f;
        if self.state == State::Release {
            self.rate_counter_period = RATE_COUNTER_PERIOD[self.release as usize];
        }
    }

    /// Step the envelope counter based on current ADSR state.
    /// Attack increments, Decay/Release decrement.
    #[inline]
    const fn step_envelope(&mut self) {
        match self.state {
            State::Attack => {
                // Counter can flip 0xff→0x00 via release→attack transition,
                // freezing at zero until another release→attack cycle.
                self.envelope_counter = self.envelope_counter.wrapping_add(1);
                if self.envelope_counter == 0xff {
                    self.state = State::DecaySustain;
                    self.rate_counter_period = RATE_COUNTER_PERIOD[self.decay as usize];
                }
            }
            State::DecaySustain => {
                if self.envelope_counter != SUSTAIN_LEVEL[self.sustain as usize] {
                    self.envelope_counter = self.envelope_counter.wrapping_sub(1);
                }
            }
            State::Release => {
                // Counter can flip 0x00→0xff via attack→release transition,
                // then continues counting down.
                self.envelope_counter = self.envelope_counter.wrapping_sub(1);
            }
        }
    }

    /// Update exponential counter period based on envelope counter value.
    /// Period increases as counter decreases, modeling RC discharge curve.
    #[inline]
    const fn update_exponential_period(&mut self) {
        match self.envelope_counter {
            0xff => self.exponential_counter_period = 1,
            0x5d => self.exponential_counter_period = 2,
            0x36 => self.exponential_counter_period = 4,
            0x1a => self.exponential_counter_period = 8,
            0x0e => self.exponential_counter_period = 16,
            0x06 => self.exponential_counter_period = 30,
            0x00 => {
                self.exponential_counter_period = 1;
                // Counter frozen at zero until gate cycles off→on.
                self.hold_zero = true;
            }
            _ => {}
        }
    }

    #[inline]
    /// Clock the envelope generator by one SID cycle.
    pub fn clock(&mut self) {
        // ADSR delay bug: if rate_counter_period is set below rate_counter,
        // counter wraps at 2^15 before envelope can step.
        self.rate_counter += 1;
        if self.rate_counter & RATE_COUNTER_MSB_MASK != 0 {
            self.rate_counter += 1;
            self.rate_counter &= RATE_COUNTER_MASK;
        }
        if self.rate_counter != self.rate_counter_period {
            return;
        }
        self.rate_counter = 0;

        // Attack state resets exponential counter on first step.
        self.exponential_counter += 1;
        if self.state != State::Attack
            && self.exponential_counter != self.exponential_counter_period
        {
            return;
        }
        self.exponential_counter = 0;

        if self.hold_zero {
            return;
        }
        self.step_envelope();
        self.update_exponential_period();
    }

    #[inline]
    /// Clock the envelope by multiple cycles, honoring rate counter boundaries.
    pub fn clock_delta(&mut self, mut delta: u32) {
        // Calculate cycles until next rate counter match (two's complement math).
        let mut rate_step = self.rate_counter_period as i32 - self.rate_counter as i32;
        if rate_step <= 0 {
            rate_step += 0x7fff;
        }

        while delta != 0 {
            if delta < rate_step as u32 {
                // Partial step: just advance rate counter
                self.rate_counter += delta as u16;
                if self.rate_counter & RATE_COUNTER_MSB_MASK != 0 {
                    self.rate_counter += 1;
                    self.rate_counter &= RATE_COUNTER_MASK;
                }
                return;
            }

            // Full rate period elapsed
            self.rate_counter = 0;
            delta -= rate_step as u32;

            // Attack state resets exponential counter on first step.
            self.exponential_counter += 1;
            if self.state == State::Attack
                || self.exponential_counter == self.exponential_counter_period
            {
                self.exponential_counter = 0;
                if !self.hold_zero {
                    self.step_envelope();
                    self.update_exponential_period();
                }
            }
            rate_step = self.rate_counter_period as i32;
        }
    }

    #[inline]
    /// Current envelope output level (0-255).
    pub const fn output(&self) -> u8 {
        self.envelope_counter
    }

    /// Alias for `output`, used by register reads.
    pub const fn read_env(&self) -> u8 {
        self.envelope_counter
    }

    /// Reset to initial state (Release, counters zeroed).
    pub const fn reset(&mut self) {
        self.attack = 0;
        self.decay = 0;
        self.sustain = 0;
        self.release = 0;
        self.gate = false;
        self.state = State::Release;
        self.envelope_counter = 0;
        self.exponential_counter = 0;
        self.exponential_counter_period = 1;
        self.hold_zero = true;
        self.rate_counter = 0;
        self.rate_counter_period = RATE_COUNTER_PERIOD[self.release as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_envelope() -> EnvelopeGenerator {
        let mut gen = EnvelopeGenerator::default();
        gen.reset();
        gen.envelope_counter = 0;
        gen
    }

    fn clock_n(gen: &mut EnvelopeGenerator, n: u32) {
        for _ in 0..n {
            gen.clock();
        }
    }

    /// ADSR delay bug: lowering attack rate mid-envelope causes rate counter
    /// to wrap around 0x8000 before the next step occurs.
    #[test]
    fn adsr_delay_bug() {
        let mut gen = new_envelope();
        gen.set_attack_decay(0x70);
        gen.set_control(0x01);
        clock_n(&mut gen, 200);

        assert_eq!(gen.read_env(), 0);

        gen.set_attack_decay(0x20);
        clock_n(&mut gen, 200);

        assert_eq!(
            gen.read_env(),
            0,
            "ADSR delay bug: counter must wrap 0x8000"
        );
    }

    /// Counter wraps 0xff->0x00 via release->attack transition, then freezes.
    #[test]
    fn flip_ff_to_00() {
        let mut gen = new_envelope();
        gen.set_attack_decay(0x77);
        gen.set_sustain_release(0x77);
        gen.set_control(0x01);

        while gen.read_env() != 0xff {
            gen.clock();
        }

        gen.set_control(0x00);
        clock_n(&mut gen, 3);
        gen.set_control(0x01);
        clock_n(&mut gen, 315);

        assert_eq!(
            gen.read_env(),
            0,
            "Counter should wrap 0xff->0x00 and freeze"
        );
    }

    /// Counter wraps 0x00->0xff via attack->release transition.
    #[test]
    fn flip_00_to_ff() {
        let mut gen = new_envelope();
        gen.hold_zero = true;
        gen.set_attack_decay(0x77);
        gen.set_sustain_release(0x77);
        gen.clock();

        assert_eq!(gen.read_env(), 0);

        gen.set_control(0x01);
        clock_n(&mut gen, 3);
        gen.set_control(0x00);
        clock_n(&mut gen, 315);

        assert_eq!(gen.read_env(), 0xff, "Counter should wrap 0x00->0xff");
    }

    macro_rules! test_attack_rate {
        ($name:ident, $attack:expr, $period:expr) => {
            #[test]
            fn $name() {
                let mut gen = new_envelope();
                gen.set_attack_decay($attack << 4);
                gen.set_control(0x01);

                let mut cycles = 0u32;
                while gen.read_env() == 0 && cycles < 100_000 {
                    gen.clock();
                    cycles += 1;
                }

                assert!(
                    cycles <= $period + 10,
                    "Attack {} period: expected ~{}, got {}",
                    $attack,
                    $period,
                    cycles
                );
            }
        };
    }

    test_attack_rate!(attack_rate_0, 0, 9);
    test_attack_rate!(attack_rate_1, 1, 32);
    test_attack_rate!(attack_rate_2, 2, 63);
}
