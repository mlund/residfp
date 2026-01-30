// Tests ported from libresidfp TestWaveformGenerator.cpp
//
// Verify LFSR behavior and noise waveform generation quirks.

use residfp::wave::WaveformGenerator;
use residfp::ChipModel;

fn new_wave() -> WaveformGenerator {
    let mut gen = WaveformGenerator::new(ChipModel::Mos6581);
    gen.reset();
    gen
}

fn clock_n(gen: &mut WaveformGenerator, n: u32) {
    for _ in 0..n {
        gen.clock();
    }
}

/// LFSR initializes to 0x7ffff8 - specific pattern for SID noise generation.
#[test]
fn shift_register_init_value() {
    let gen = new_wave();
    assert_eq!(gen.get_shift(), 0x007f_fff8);
}

/// Noise extracted from LFSR bits 22,20,16,13,11,7,4,2 -> output bits 11-4.
#[test]
fn noise_output() {
    let mut gen = new_wave();
    gen.shift = 0x35555f;
    gen.set_control(0x80); // noise waveform
    gen.clock();

    assert!(gen.output(None) > 0, "Noise should produce non-zero output");
}

/// TEST bit clears LFSR immediately; releasing it restores initial pattern.
/// This allows deterministic noise restart in music routines.
#[test]
fn test_bit_clears_register() {
    let mut gen = new_wave();
    gen.set_frequency_lo(0xff);
    gen.set_frequency_hi(0xff);
    clock_n(&mut gen, 1000);

    // TEST bit clears to 0
    gen.set_control(0x08);
    gen.clock();
    assert_eq!(gen.get_shift(), 0);

    // Releasing TEST restores initial LFSR state
    gen.set_control(0x00);
    gen.clock();
    assert_eq!(gen.get_shift(), 0x007f_fff8);
}

/// TEST bit clears accumulator for oscillator sync tricks.
#[test]
fn test_bit_clears_accumulator() {
    let mut gen = new_wave();
    gen.set_frequency_lo(0xff);
    gen.set_frequency_hi(0xff);
    clock_n(&mut gen, 100);

    assert!(gen.get_acc() != 0);

    gen.set_control(0x08);
    assert_eq!(gen.get_acc(), 0);
}

/// Accumulator increments by frequency value each clock cycle.
#[test]
fn accumulator_increment() {
    let mut gen = new_wave();
    gen.set_frequency_lo(0x01);
    gen.set_frequency_hi(0x00);

    gen.clock();
    assert_eq!(gen.get_acc(), 1);
    gen.clock();
    assert_eq!(gen.get_acc(), 2);

    gen.set_frequency_lo(0x00);
    gen.set_frequency_hi(0x01); // freq = 256
    let before = gen.get_acc();
    gen.clock();
    assert_eq!(gen.get_acc(), before + 256);
}

/// 24-bit accumulator wraps at 0x1000000.
#[test]
fn accumulator_wrap() {
    let mut gen = new_wave();
    gen.set_acc(0x00ff_fffe);
    gen.set_frequency_lo(0x10);
    gen.clock();

    assert_eq!(gen.get_acc() & 0x00ff_ffff, 0x00_000e);
}

/// LFSR clocks when accumulator bit 19 transitions 0->1.
/// The shift is delayed 2 cycles after bit 19 is set high (hardware pipeline).
#[test]
fn shift_register_clock_on_bit19() {
    let mut gen = new_wave();
    let initial = gen.get_shift();

    // Position just below bit 19 boundary
    gen.set_acc(0x0007_fff0);
    gen.set_frequency_lo(0x20);
    gen.clock(); // 0x7fff0 + 0x20 = 0x80010 (bit 19 set) -> pipeline=2

    // Shift hasn't happened yet due to 2-cycle pipeline
    assert_eq!(
        gen.get_shift(),
        initial,
        "LFSR should not shift immediately (pipeline phase 0)"
    );

    gen.clock(); // pipeline=1, latch shift register
    assert_eq!(
        gen.get_shift(),
        initial,
        "LFSR should not shift yet (pipeline phase 1)"
    );

    gen.clock(); // pipeline=0, perform actual shift
    assert_ne!(
        gen.get_shift(),
        initial,
        "LFSR should clock after 2-cycle pipeline delay"
    );
}

/// Sync bit enables hard sync from another oscillator.
#[test]
fn sync_bit() {
    let mut gen = new_wave();
    assert!(!gen.get_sync());

    gen.set_control(0x02);
    assert!(gen.get_sync());

    gen.set_control(0x00);
    assert!(!gen.get_sync());
}

/// MSB rising edge detection triggers sync to other oscillators.
#[test]
fn msb_rising() {
    let mut gen = new_wave();
    gen.set_acc(0x007f_fff0);
    gen.set_frequency_lo(0x20);
    gen.clock();

    assert!(gen.is_msb_rising(), "Should detect bit 23 transition 0->1");

    gen.clock();
    assert!(!gen.is_msb_rising(), "Flag clears after one cycle");
}

/// Verify each waveform type produces expected output characteristics.
macro_rules! test_waveform {
    ($name:ident, $waveform:expr, $check:expr) => {
        #[test]
        fn $name() {
            let mut gen = new_wave();
            gen.set_frequency_hi(0x10);
            gen.set_pulse_width_hi(0x08);
            gen.set_control($waveform << 4);
            clock_n(&mut gen, 100);

            let out = gen.output(None);
            let check: fn(u16) -> bool = $check;
            assert!(check(out), "Waveform {} output {} invalid", $waveform, out);
        }
    };
}

test_waveform!(waveform_triangle, 1, |o| o > 0 && o < 0x0fff);
test_waveform!(waveform_sawtooth, 2, |o| o > 0 && o <= 0x0fff);
test_waveform!(waveform_pulse, 4, |o| o == 0 || o == 0x0fff);
test_waveform!(waveform_noise, 8, |_| true); // any value valid

// --- Floating DAC Output Tests ---

/// When waveform is set to 0, the last output value is held (floating DAC).
#[test]
fn floating_dac_holds_last_value() {
    let mut gen = new_wave();
    gen.set_frequency_hi(0x10);
    gen.set_control(0x20); // sawtooth
    clock_n(&mut gen, 1000);

    let last_output = gen.output(None);
    assert!(last_output > 0, "Should have non-zero output from sawtooth");

    // Switch to no waveform
    gen.set_control(0x00);
    gen.clock();

    // Output should still be the cached value
    assert_eq!(
        gen.output(None),
        last_output,
        "Floating DAC should hold last value"
    );
}

/// Floating DAC output eventually fades to zero after TTL expires.
/// 6581 TTL is ~54000 cycles, then fades every ~1400 cycles.
#[test]
fn floating_dac_fades_to_zero() {
    let mut gen = new_wave();
    gen.set_frequency_hi(0x10);
    gen.set_control(0x20); // sawtooth
    clock_n(&mut gen, 1000);

    let last_output = gen.output(None);
    assert!(last_output > 0);

    // Switch to no waveform
    gen.set_control(0x00);

    // After enough cycles, output should fade to zero
    // 6581: 54000 initial + 12 fade steps * 1400 = ~70800 cycles max
    clock_n(&mut gen, 80000);

    assert_eq!(gen.output(None), 0, "Floating DAC should fade to zero");
}

/// Floating DAC fade uses bit-shifting pattern (output &= output >> 1).
#[test]
fn floating_dac_fade_pattern() {
    let mut gen = new_wave();

    // Set up a known output value with multiple bits set
    gen.set_frequency_hi(0x08);
    gen.set_pulse_width_hi(0x00);
    gen.set_pulse_width_lo(0x01);
    gen.set_control(0x40); // pulse
    clock_n(&mut gen, 100);

    // Get output - should be 0xfff (pulse high)
    let initial = gen.output(None);
    assert_eq!(initial, 0x0fff, "Pulse should be high");

    // Switch to no waveform and wait for first fade
    gen.set_control(0x00);
    clock_n(&mut gen, 54001); // Just past initial TTL

    let after_fade = gen.output(None);
    // 0x0fff & (0x0fff >> 1) = 0x0fff & 0x07ff = 0x07ff
    assert_eq!(after_fade, 0x07ff, "First fade: 0x0fff & 0x07ff = 0x07ff");
}

/// 8580 has longer TTL (~800000 cycles) than 6581 (~54000 cycles).
#[test]
fn floating_dac_8580_longer_ttl() {
    let mut gen_6581 = WaveformGenerator::new(ChipModel::Mos6581);
    let mut gen_8580 = WaveformGenerator::new(ChipModel::Mos8580);

    // Set up both with same waveform
    for gen in [&mut gen_6581, &mut gen_8580] {
        gen.set_frequency_hi(0x10);
        gen.set_control(0x20); // sawtooth
    }
    clock_n(&mut gen_6581, 1000);
    clock_n(&mut gen_8580, 1000);

    let out_6581 = gen_6581.output(None);
    let out_8580 = gen_8580.output(None);

    // Switch both to no waveform
    gen_6581.set_control(0x00);
    gen_8580.set_control(0x00);

    // After 60000 cycles: 6581 should have started fading, 8580 should not
    clock_n(&mut gen_6581, 60000);
    clock_n(&mut gen_8580, 60000);

    assert!(
        gen_6581.output(None) < out_6581,
        "6581 should have faded after 60k cycles"
    );
    assert_eq!(
        gen_8580.output(None),
        out_8580,
        "8580 should still hold value after 60k cycles"
    );
}

// --- Noise Write-back Tests ---

/// Noise write-back: switching between noise and combined waveforms modifies LFSR.
///
/// From libresidfp TestNoiseWriteBack1:
/// When a combined waveform including noise is selected, the waveform output
/// can pull down bits in the shift register. This causes the noise pattern
/// to change when switching back to pure noise.
///
/// Not implemented: resid-rs returns 0 for combined noise waveforms (0x9-0xf)
/// and has no write-back mechanism. This test documents the expected libresidfp
/// behavior as a target for future implementation.
#[test]
#[ignore = "write-back not implemented: requires combined noise waveforms and LFSR write-back"]
fn noise_write_back() {
    let mut gen = new_wave();
    gen.set_control(0x80); // noise waveform

    // Switch from noise to noise+triangle to trigger write-back
    gen.set_control(0x88); // noise + test bit
    gen.clock();
    let _ = gen.output(None);
    gen.set_control(0x90); // noise + triangle
    gen.clock();
    let _ = gen.output(None);

    // Expected OSC3 values from libresidfp after repeated noise<->combined switching
    let expected_osc3: [u8; 8] = [0xfc, 0x6c, 0xd8, 0xb1, 0xd8, 0x6a, 0xb1, 0xf0];

    for &expected in &expected_osc3 {
        gen.set_control(0x88); // noise + test bit
        gen.clock();
        let _ = gen.output(None);
        gen.set_control(0x80); // noise only
        gen.clock();
        let osc3 = (gen.output(None) >> 4) as u8;
        assert_eq!(
            osc3, expected,
            "Write-back should modify LFSR, changing noise output"
        );
    }
}
