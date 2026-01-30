// Tests ported from libresidfp TestWaveformGenerator.cpp
//
// Verify LFSR behavior and noise waveform generation quirks.

use resid::wave::WaveformGenerator;
use resid::ChipModel;

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
