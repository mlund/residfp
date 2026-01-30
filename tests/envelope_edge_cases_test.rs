// Tests ported from libresidfp TestEnvelopeGenerator.cpp
//
// Verify hardware quirks in the SID's ADSR envelope generator.

use resid::envelope::EnvelopeGenerator;

/// Creates a fresh envelope generator with counter at 0.
fn new_envelope() -> EnvelopeGenerator {
    let mut gen = EnvelopeGenerator::default();
    gen.reset();
    gen.envelope_counter = 0;
    gen
}

/// Clocks the generator n times.
fn clock_n(gen: &mut EnvelopeGenerator, n: u32) {
    for _ in 0..n {
        gen.clock();
    }
}

/// Tests the ADSR delay bug: lowering attack rate mid-envelope causes
/// rate counter to wrap around 0x8000 before the next step occurs.
#[test]
fn adsr_delay_bug() {
    let mut gen = new_envelope();

    // Start with slow attack (period=313), advance 200 cycles
    gen.set_attack_decay(0x70);
    gen.set_control(0x01);
    clock_n(&mut gen, 200);

    // Not enough cycles for first increment yet
    assert_eq!(gen.read_env(), 0);

    // Lower attack rate (period=63) while rate_counter is at 200
    // Bug: counter > period, so must wrap around 0x8000 first
    gen.set_attack_decay(0x20);
    clock_n(&mut gen, 200);

    // Still stuck at 0 due to wraparound requirement
    assert_eq!(gen.read_env(), 0, "ADSR delay bug: counter must wrap 0x8000");
}

/// Counter wraps 0xff->0x00 via release->attack transition, then freezes.
/// This quirk allows "stuck envelope" scenarios in real SID chips.
#[test]
fn flip_ff_to_00() {
    let mut gen = new_envelope();
    gen.set_attack_decay(0x77);
    gen.set_sustain_release(0x77);
    gen.set_control(0x01);

    // Ramp to max
    while gen.read_env() != 0xff {
        gen.clock();
    }

    // Quick release->attack transition (3 cycles for state pipeline)
    gen.set_control(0x00);
    clock_n(&mut gen, 3);
    gen.set_control(0x01);
    clock_n(&mut gen, 315);

    // Counter wrapped and froze at 0
    assert_eq!(gen.read_env(), 0, "Counter should wrap 0xff->0x00 and freeze");
}

/// Counter wraps 0x00->0xff via attack->release transition.
/// Envelope then continues decrementing in release state.
#[test]
fn flip_00_to_ff() {
    let mut gen = new_envelope();
    gen.hold_zero = true; // Keep counter frozen at 0
    gen.set_attack_decay(0x77);
    gen.set_sustain_release(0x77);
    gen.clock();

    assert_eq!(gen.read_env(), 0);

    // Quick attack->release transition (3 cycles for state pipeline)
    gen.set_control(0x01);
    clock_n(&mut gen, 3);
    gen.set_control(0x00);
    clock_n(&mut gen, 315);

    // Counter wrapped from 0x00 to 0xff
    assert_eq!(gen.read_env(), 0xff, "Counter should wrap 0x00->0xff");
}

/// Verify attack rate timing matches hardware-measured periods.
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
                $attack, $period, cycles
            );
        }
    };
}

// Rate counter periods from SID Programmer's Reference Guide
test_attack_rate!(attack_rate_0, 0, 9);   // 2ms
test_attack_rate!(attack_rate_1, 1, 32);  // 8ms
test_attack_rate!(attack_rate_2, 2, 63);  // 16ms
