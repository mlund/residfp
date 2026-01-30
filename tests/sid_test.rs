mod data;

use resid::{ChipModel, Sid};

#[rustfmt::skip]
static SID_DATA: [u16; 51] = [
    25, 177, 250, 28, 214, 250,
    25, 177, 250, 25, 177, 250,
    25, 177, 125, 28, 214, 125,
    32, 94, 750, 25, 177, 250,
    28, 214, 250, 19, 63, 250,
    19, 63, 250, 19, 63, 250,
    21, 154, 63, 24, 63, 63,
    25, 177, 250, 24, 63, 125,
    19, 63, 250,
];

/// Generate SID output sequence for golden data comparison.
/// Uses note sequence from SID_DATA with attack/decay envelope.
fn generate_sid_output() -> Vec<i16> {
    let mut sid = Sid::new(ChipModel::Mos6581);
    sid.write(0x05, 0x09); // AD1
    sid.write(0x06, 0x00); // SR1
    sid.write(0x18, 0x0f); // MODVOL

    let mut outputs = Vec::new();
    let mut i = 0;
    while i < SID_DATA.len() {
        sid.write(0x01, SID_DATA[i + 0] as u8); // FREQHI1
        sid.write(0x00, SID_DATA[i + 1] as u8); // FREQLO1
        sid.write(0x00, 0x21); // CR1
        for _ in 0..SID_DATA[i + 2] {
            sid.clock_delta(22);
            outputs.push(sid.output());
        }
        sid.write(0x00, 0x20); // CR1
        for _ in 0..50 {
            sid.clock_delta(22);
            outputs.push(sid.output());
        }
        i += 3;
    }
    outputs
}

#[test]
fn clock_delta() {
    let outputs = generate_sid_output();
    let expected = &data::sid_output::RESID_OUTPUT;

    assert_eq!(outputs.len(), expected.len(), "Output length mismatch");

    for (i, (&got, &exp)) in outputs.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got, exp, "Mismatch at index {}", i);
    }
}

#[test]
#[ignore = "Run manually to regenerate golden data"]
fn generate_golden_data() {
    let outputs = generate_sid_output();

    println!("#[allow(unused)]");
    println!("#[rustfmt::skip]");
    println!("pub static RESID_OUTPUT: [i16; {}] = [", outputs.len());
    for chunk in outputs.chunks(8) {
        print!("    ");
        print!(
            "{}",
            chunk
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(",");
    }
    println!("];");
}
