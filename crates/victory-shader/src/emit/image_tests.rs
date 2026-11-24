//! End-to-end tests of synthetic GFX10 image samples lowered to DXIL,
//! validated by `dxil.dll` when present, and run on the host GPU and WARP.

use super::host::{Dispatch, Host, ONE_AT_A_TIME, Root, View};
use super::{HostShader, StaticSampler, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A 1-thread compute shader with 4 root constants in `s0..s3` and a
/// descriptor table in `s4`: heap entry 0 is a cube image, entry 1 a raw
/// UAV. It samples level 0 at the centre of a face (s = t = 1.5) with the
/// face operand `face` leaves in `v3`, and stores the red channel to
/// dword 0 of the UAV.
fn cube_cs(face: &[u32]) -> HostShader {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], 0
        0xFA00_0000,
        0xF408_0502, // s_load_dwordx4 s[20:23], s[4:5], 32
        0xFA00_0020,
        0xBE90_0380, // s_mov_b32 s16, 0: point sampling, wrap
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE93_0380, // s_mov_b32 s19, 0
        0x7E02_02FF, // v_mov_b32 v1, 1.5
        0x3FC0_0000,
        0x7E04_02FF, // v_mov_b32 v2, 1.5
        0x3FC0_0000,
    ];
    words.extend_from_slice(face);
    words.extend_from_slice(&[
        0xF09C_0118, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:0x1 dim:CUBE
        0x0082_0401,
        0xE070_0000, // buffer_store_dword v4, off, s[20:23], 0
        0x8005_0400,
        S_ENDPGM,
    ]);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let table = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![
            table(ResourceType::Srv, 0, 8),
            table(ResourceType::Uav, 8, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// 1×1 faces whose red channel is 10 × the cube + the face (+x -x +y -y
/// +z -z).
fn faces(cubes: u32) -> Vec<[f32; 4]> {
    (0..cubes * 6)
        .map(|n| [(10 * (n / 6) + n % 6) as f32, 0.0, 0.0, 1.0])
        .collect()
}

/// Runs `shader` with `view` at heap entry 0 on every available SM 6.6
/// host (hardware GPU and WARP), once per case: the root constants s0 and
/// s1, and the red channel the sample should read.
fn check(shader: &HostShader, view: View, cases: &[([f32; 2], f32)]) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let views = [view, View::RawUav(1)];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut failures = Vec::new();
        for &(inputs, want) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot, u.dword) {
                    (RootType::Constants, 0, d @ 0..=1) => inputs[d as usize].to_bits(),
                    (RootType::Table, 1, _) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got = f32::from_bits(out[0][0]);
            if got != want {
                failures.push(format!("s0, s1 = {inputs:?}: read {got}, want {want}"));
            }
        }
        assert!(failures.is_empty(), "{label}:\n{}", failures.join("\n"));
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Every (cube, face) of a 2-cube array.
fn every_face() -> Vec<([f32; 2], f32)> {
    (0..2u8)
        .flat_map(|cube| (0..6u8).map(move |face| (cube, face)))
        .map(|(cube, face)| {
            let want = f32::from(10 * cube + face);
            ([f32::from(cube), f32::from(face)], want)
        })
        .collect()
}

#[test]
fn cube_array_samples_take_the_slice_from_the_face_operand() {
    // How titles build a cube array's face operand: 8 × the slice + the
    // face id.
    let shader = cube_cs(&[
        0x7E06_0201, // v_mov_b32 v3, s1 (face id)
        0x7E0A_0200, // v_mov_b32 v5, s0 (slice)
        0x5606_0AFF, // v_fmac_f32 v3, 0x41000000 (8.0), v5
        0x4100_0000,
    ]);
    let texels = faces(2);
    check(&shader, View::CubeArray(1, 2, &texels), &every_face());
}

#[test]
fn cube_array_samples_take_the_slice_from_a_constant_face_operand() {
    // 10.0: slice 1, face 2 (+y).
    let shader = cube_cs(&[
        0x7E06_02FF, // v_mov_b32 v3, 10.0
        0x4120_0000,
    ]);
    let texels = faces(2);
    check(&shader, View::CubeArray(1, 2, &texels), &[([0.0; 2], 12.0)]);
}

#[test]
fn cube_samples_take_a_bare_face_operand_as_a_cube() {
    let shader = cube_cs(&[
        0x7E06_0201, // v_mov_b32 v3, s1 (face id)
    ]);
    let texels = faces(1);
    let cases: Vec<_> = (0..6u8)
        .map(|face| ([0.0, f32::from(face)], f32::from(face)))
        .collect();
    check(&shader, View::Cube(1, &texels), &cases);
}
