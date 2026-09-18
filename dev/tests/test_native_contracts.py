from __future__ import annotations

import math
import re
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLANG = ROOT / "native" / "slang"


class NativeContractTests(unittest.TestCase):
    PIPELINES = {
        "ftgspp_materialize": 6,
        "ftgspp_fused_backward_optimizer": 19,
        "ftgspp_fused_dynamic_backward_adam": 21,
        "ftgspp_explicit_adam": 21,
        "ftgspp_relocation_prepare": 9,
        "ftgspp_relocation_copy": 14,
        "ftgspp_flow_features": 4,
        "ftgspp_loss_reduce": 6,
        "ftgspp_eval_metrics": 6,
        "ftgspp_color_correct_forward": 4,
        "ftgspp_color_correct_backward": 6,
        "ftgspp_color_adam": 6,
        "ftgspp_tile_work_prepare": 4,
    }

    def test_shader_bindings_are_contiguous_and_match_cpp_pipeline(self):
        header = (ROOT / "native" / "src" / "training" / "gs_trainer.h").read_text(encoding="utf-8")
        header += (ROOT / "native" / "src" / "render" / "gs_renderer.h").read_text(encoding="utf-8")
        for name, count in self.PIPELINES.items():
            source = (SLANG / f"{name}.slang").read_text(encoding="utf-8")
            bindings = sorted(
                int(value)
                for value in re.findall(r"layout\s*\(\s*binding\s*=\s*(\d+)\s*\)", source)
            )
            self.assertEqual(bindings, list(range(count)), name)
            declaration = re.compile(
                rf"pipeline_{re.escape(name)}\s*=\s*_ComputePipeline\({count}\)"
            )
            self.assertRegex(header, declaration, name)

    def test_multi_phase_relocation_descriptors(self):
        header = (ROOT / "native" / "src" / "training" / "gs_trainer.h").read_text(encoding="utf-8")
        expected = {
            "ftgspp_relocation_score": 5,
            "ftgspp_relocation_cdf": 3,
            "ftgspp_relocation_cdf_normalize": 2,
            "ftgspp_relocation_sample": 5,
        }
        for name, count in expected.items():
            self.assertRegex(
                header, rf"pipeline_{name}\s*=\s*_ComputePipeline\({count}\)", name)

    def test_dynamic_push_constant_matrix_has_explicit_alignment_padding(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(encoding="utf-8")
        slang = (SLANG / "ftgspp_fused_backward_optimizer.slang").read_text(encoding="utf-8")
        self.assertRegex(
            cpp,
            r"float reg_gate;\s*float reserved\[3\];\s*"
            r"float world_view_transform\[16\]",
        )
        self.assertRegex(
            slang,
            r"float reg_gate;\s*float reserved0;\s*float reserved1;\s*"
            r"float reserved2;\s*"
            r"float4x4 world_view_transform",
        )

    def test_projection_cull_reason_and_radius_outputs_stay_consistent(self):
        shader = (SLANG / "vertex_shader.slang").read_text(encoding="utf-8")
        renderer = (ROOT / "native" / "src" / "render" / "gs_renderer.cpp").read_text(
            encoding="utf-8"
        )
        header = (ROOT / "native" / "src" / "render" / "gs_renderer.h").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "layout(binding=11) RWStructuredBuffer<uint> out_cull_reason;",
            shader,
        )
        self.assertIn("out_cull_reason[g_idx] = (1u << 5);", shader)
        ellipse_zero = shader.index(
            "out_cull_reason[g_idx] = (1u << 5);"
        )
        radius_write = shader.index(
            "out_radii[g_idx] = (uint32_t)ceil", ellipse_zero
        )
        self.assertLess(ellipse_zero, radius_write)
        self.assertIn(
            "resizeDeviceBuffer(buffers.projection_cull_reason, alloc_size)",
            renderer,
        )
        self.assertIn(
            "pipeline_projection_forward = _ComputePipeline(12)", header,
        )
        self.assertIn(
            ".DEF_BUFFER_ARRAY(projection_cull_reason, uint32_t", bindings,
        )

    def test_exact_tile_count_and_emit_share_scanline_helper(self):
        config = (SLANG / "config.slang").read_text(encoding="utf-8")
        utils = (SLANG / "utils.slang").read_text(encoding="utf-8")
        vertex = (SLANG / "vertex_shader.slang").read_text(encoding="utf-8")
        tile = (SLANG / "tile_shader.slang").read_text(encoding="utf-8")
        self.assertIn("#define TILE_CULLING_MODE 2", config)
        self.assertIn("EllipseTileSpan ellipse_scanline_tile_span(", utils)
        self.assertIn("max(bv0*bv0 - a*(c*v0*v0-1.0f), 0.0f)", utils)
        self.assertIn("int(floor((bound.y + center.x) / range_tile_size)) + 1", utils)
        self.assertGreaterEqual(
            utils.count("ellipse_scanline_tile_span("), 3
        )
        self.assertEqual(tile.count("ellipse_scanline_tile_span("), 2)
        self.assertNotIn("offset%(grid_width*grid_height)", tile)
        self.assertNotIn("for (; offset < offset_max; offset++)", tile)
        self.assertIn(
            "layout(binding=7) RWByteAddressBuffer "
            "out_tile_invariant_status;", tile,
        )
        self.assertIn("if (offset != offset_max)", tile)
        self.assertIn("invariant_bits |= 1u", tile)
        self.assertIn("invariant_bits |= 2u", tile)
        self.assertIn("InterlockedOr(0u, invariant_bits)", tile)
        renderer_h = (ROOT / "native" / "src" / "render" / "gs_renderer.h").read_text(
            encoding="utf-8"
        )
        renderer = (ROOT / "native" / "src" / "render" / "gs_renderer.cpp").read_text(
            encoding="utf-8"
        )
        backend = (ROOT / "src" / "ftgspp" / "training" / "backend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "pipeline_generate_keys = _ComputePipeline(9)", renderer_h,
        )
        self.assertIn(
            "clearDeviceBuffer(buffers.tile_invariant_status, 1)", renderer,
        )
        self.assertIn("get_tile_invariant_status", backend)
        for output in (
            "out_tiles_touched", "out_rect_tile_space", "out_radii",
            "out_xy_vs", "out_depths", "out_inv_cov_vs", "out_rgb",
        ):
            self.assertIn(output, vertex[vertex.index("void clear_projected_outputs"):])

    def test_tensor_backward_keeps_group_barriers_uniform(self):
        shader = (SLANG / "alphablend_shader_bwd_tensor.slang").read_text(
            encoding="utf-8"
        )
        start = shader.index("const bool valid_splat")
        end = shader.index(
            "GroupMemoryBarrierWithGroupSync();",
            shader.index("for (int s =", start),
        )
        self.assertNotIn("continue;", shader[start:end])
        self.assertIn(
            "pixel_batch_rank < GROUP_REDUCE_BEFORE_ATOMIC && valid_splat",
            shader,
        )

    def test_sh_zero_clamp_keeps_gsplat_boundary_gradient(self):
        shader = (SLANG / "spherical_harmonics.slang").read_text(
            encoding="utf-8"
        )
        start = shader.index("float3 compute_color_from_sh_coeffs")
        body = shader[start : shader.index("\n}", start) + 2]
        self.assertNotIn("max(rgb, float(0.0))", body)
        for channel in "xyz":
            self.assertIn(
                f"rgb.{channel} = rgb.{channel} < 0.0 ? 0.0 : rgb.{channel};",
                body,
            )

    def test_flow_feature_overwrite_is_ordered_after_projection(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(encoding="utf-8")
        start = cpp.index("void VulkanGSTrainer::executeFTGSPPFlowFeatures")
        body = cpp[start : cpp.index("\n}\n", start) + 3]
        self.assertIn("xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE", body)
        self.assertIn("rgb.deviceBuffer, COMPUTE_SHADER_WRITE", body)
        self.assertIn("COMPUTE_SHADER_READ_WRITE", body)

    def test_python_runtime_has_no_sibling_repository_imports(self):
        forbidden = ("import gsplat", "import vksplat", "import tinycudann", "sys.path.insert")
        sources = list((ROOT / "src" / "ftgspp").rglob("*.py")) + list(
            (ROOT / "scripts").rglob("*.py")
        )
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} in {path.relative_to(ROOT)}")

    def test_native_dynamic_view_is_one_call_per_training_step(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(encoding="utf-8")
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(encoding="utf-8")
        self.assertIn('def("train_dynamic_view"', bindings)
        self.assertIn("backend.train_view(", runner)
        self.assertIn(
            "trainer.executeComputeSSIMGradient(\n"
            "            config, uniforms, buffers, view_idx,",
            bindings,
        )

    def test_fused_optimizer_respects_normalize_quats_flag(self):
        config = (SLANG / "config.slang").read_text(encoding="utf-8")
        adam = (SLANG / "ftgspp_explicit_adam.slang").read_text(encoding="utf-8")
        self.assertIn("#define NORMALIZE_QUATS_IN_OPTIMIZER 0", config)
        self.assertIn(
            "rotations[idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;",
            adam,
        )

    def test_explicit_adam_can_log_geometry_gradient_stats(self):
        adam = (SLANG / "ftgspp_explicit_adam.slang").read_text(encoding="utf-8")
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("layout(binding=19) RWByteAddressBuffer grad_stats;", adam)
        self.assertIn("if (uniforms.reserved2 > 0.5f)", adam)
        self.assertIn("const float rot_sum_sq = dot(g_rot_t, g_rot_t) * q_norm_sq;", adam)
        self.assertIn("InterlockedAddF32(16u, sh_grad_sum_sq.x)", adam)
        self.assertIn("InterlockedAddF32(20u, sh_grad_sum_sq.y)", adam)
        self.assertIn("u.reserved[2] = collect_geometry_grad_stats ? 1.0f : 0.0f;", trainer)
        self.assertIn("clearDeviceBuffer(buffers.grad_stats, 6);", trainer)
        self.assertIn("get_geometry_gradient_stats", bindings)
        self.assertIn("set_collect_geometry_grad_stats", bindings)
        self.assertIn('result["grad_sh_dc_rms"]', bindings)
        self.assertIn('result["grad_sh_rest_rms"]', bindings)

    def test_native_loss_scalar_covers_the_optimized_objective(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        fused = (SLANG / "ftgspp_fused_backward_optimizer.slang").read_text(
            encoding="utf-8"
        )
        color = (SLANG / "ftgspp_color_adam.slang").read_text(encoding="utf-8")
        for key in (
            "l1", "ssim", "reg_opacity", "reg_scale", "gate", "color_cc",
            "image_loss", "total_loss",
        ):
            self.assertIn(f'result["{key}"]', bindings)
        for offset in ("8u", "12u", "16u"):
            self.assertIn(f"InterlockedAddF32({offset}", fused)
        self.assertIn("InterlockedAddF32(20u", color)
        self.assertIn("u.reg * color_params[i] * color_params[i]", color)
        self.assertIn(
            "buffers.loss_accum.deviceBuffer.size != 6*sizeof(float)", bindings,
        )
        self.assertGreaterEqual(
            trainer.count("clearDeviceBuffer(buffers.loss_accum, 6)"), 1
        )

    def test_ftgspp_loss_and_color_correction_share_the_exact_clamp_chain(self):
        ssim = (SLANG / "ssim.slang").read_text(encoding="utf-8")
        scalar = (SLANG / "ftgspp_loss_reduce.slang").read_text(encoding="utf-8")
        color_forward = (
            SLANG / "ftgspp_color_correct_forward.slang"
        ).read_text(encoding="utf-8")
        color_backward = (
            SLANG / "ftgspp_color_correct_backward.slang"
        ).read_text(encoding="utf-8")
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )

        objective = (SLANG / "ftgspp_image_objective.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn("uint predictionClampMode", ssim)
        self.assertIn("uint prediction_clamp_mode", scalar)
        self.assertIn("FTGSPP_OBJECTIVE_SCENE_RGB_V1", objective)
        self.assertIn("state.rgb + background * state.w", objective)
        self.assertIn("reference * mask + background", objective)
        self.assertIn("result.target = reference;", objective)
        self.assertNotIn("mask_chain", ssim)
        self.assertIn("ftgspp_clamp_derivative", ssim)
        self.assertIn("dLdimg1Buffer[pixId].w +=", ssim)
        self.assertIn("objective.prediction", scalar)
        self.assertIn("groupshared float l1_values[256];", scalar)
        self.assertIn("groupshared float ssim_values[256];", scalar)
        self.assertIn("for (uint stride = 128u; stride > 0u; stride >>= 1u)", scalar)
        self.assertIn("if (lane == 0u)", scalar)
        self.assertEqual(scalar.count("InterlockedAddF32"), 2)
        ssim_fn = trainer[
            trainer.index("void VulkanGSTrainer::executeComputeSSIMGradient(") :
            trainer.index("void VulkanGSTrainer::executeFTGSPPMaterialize(")
        ]
        self.assertEqual(ssim_fn.count("sizeof(shader_uniforms)"), 3)
        self.assertIn("shader_uniforms.objective_mode = config.image_objective", trainer)
        self.assertIn("pending_background_ = background", bindings)
        self.assertIn("executeComputeSSIMGradient(", trainer)

        self.assertIn("(1.0f-raw.w)", color_forward)
        self.assertIn("const bool composite", color_backward)
        self.assertIn("float3 dc = mul(g, transpose(mat))", color_backward)
        self.assertIn("v_pixel_state[pix+3u] -= dot(g, bias)", color_backward)
        self.assertIn("c.x*g.x", color_backward)

    def test_explicit_backward_writes_complete_raw_fp32_state(self):
        backward = (SLANG / "ftgspp_fused_backward_optimizer.slang").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("void ftgspp_adam(", backward)
        for name in (
            "raw_grad_base_xyz_ws",
            "raw_grad_base_log_scales_opacity",
            "raw_grad_times_durations",
            "raw_grad_velocities_gates",
            "raw_grad_sh_coeffs",
            "raw_grad_rotations",
        ):
            self.assertIn(name, backward)
        self.assertIn("relocation_stats[idx] +=", backward)
        self.assertNotIn("relocation_stats[idx] =", backward)

    def test_explicit_adam_consumes_then_clears_raw_gradients(self):
        adam = (SLANG / "ftgspp_explicit_adam.slang").read_text(encoding="utf-8")
        self.assertIn("void ftgspp_adam(", adam)
        self.assertIn("raw_grad_sh_coeffs[bid] = float4(0);", adam)
        self.assertIn(
            "write_t3_float3(float3(0), idx, raw_grad_base_xyz_ws);", adam
        )
        self.assertIn("velocities_gates[idx] = x;", adam)
        self.assertIn("if (uniforms.reserved0 < 0.5f)", adam)
        self.assertIn("is_sh_skipped(uniforms.step, cid, num_sh)", adam)

    def test_cpp_keeps_split_multi_gpu_and_fused_single_gpu_paths(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        fused_start = cpp.index(
            "void VulkanGSTrainer::executeFTGSPPFusedDynamicBackwardAdam("
        )
        fused = cpp[fused_start : cpp.index("\n}\n", fused_start) + 3]
        self.assertNotIn("executeFTGSPPBackward(", fused)
        self.assertNotIn("executeFTGSPPExplicitAdam(", fused)
        backward_start = cpp.index("void VulkanGSTrainer::executeFTGSPPBackward(")
        backward = cpp[backward_start : cpp.index("\n}\n", backward_start) + 3]
        self.assertIn("v_xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE", backward)
        for name in (
            "grad_base_xyz_ws",
            "grad_base_log_scales_opacity",
            "grad_times_durations",
            "grad_velocities_gates",
            "grad_sh_coeffs",
            "grad_rotations",
            "relocation_stats",
        ):
            self.assertIn(f"{name}.deviceBuffer, COMPUTE_SHADER_WRITE", backward)
        self.assertIn("TRANSFER_COMPUTE_SHADER_READ_WRITE", backward)

        adam_start = cpp.index("void VulkanGSTrainer::executeFTGSPPExplicitAdam(")
        adam = cpp[adam_start : cpp.index("\n}\n", adam_start) + 3]
        for name in (
            "grad_base_xyz_ws",
            "grad_base_log_scales_opacity",
            "grad_times_durations",
            "grad_velocities_gates",
            "grad_sh_coeffs",
            "grad_rotations",
        ):
            self.assertIn(
                f"{name}.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE", adam
            )
        self.assertIn("COMPUTE_SHADER_READ_WRITE", adam)

    def test_single_fused_shader_has_no_raw_gradient_storage_boundary(self):
        fused = (
            SLANG / "ftgspp_fused_dynamic_backward_adam.slang"
        ).read_text(encoding="utf-8")
        for name in (
            "raw_grad_base_xyz_ws", "raw_grad_base_log_scales_opacity",
            "raw_grad_times_durations", "raw_grad_velocities_gates",
            "raw_grad_sh_coeffs", "raw_grad_rotations",
        ):
            self.assertNotIn(name, fused)
        self.assertIn("RWStructuredBuffer<float> base_xyz_ws", fused)
        self.assertIn("ftgspp_fused_adam(", fused)

        timer = (ROOT / "native" / "src" / "device" / "perf_timer.h").read_text(
            encoding="utf-8"
        )
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("_(FusedDynamicBackwardAdam)", timer)
        start = trainer.index(
            "void VulkanGSTrainer::executeFTGSPPFusedDynamicBackwardAdam("
        )
        body = trainer[start : trainer.index("\n}\n", start) + 3]
        self.assertIn(
            "Timer<PerfTimer::FusedDynamicBackwardAdam> timer(this)", body,
        )
        self.assertNotIn("Timer<PerfTimer::DynamicBackward>", body)
        self.assertNotIn("Timer<PerfTimer::ExplicitAdam>", body)

    def test_sparse_v2_uses_active_sh_and_runtime_auto_exact_choice(self):
        sparse = (SLANG / "ftgspp_sparse_gradient.slang").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("return 18u + active_sh_float_count(active_sh)", sparse)
        self.assertIn("const uint sh_floats = active_sh_float_count", sparse)
        self.assertIn("size_t sparse_record_words() const", bindings)
        self.assertIn("active_sh_float_count(uniforms.active_sh)", bindings)
        self.assertIn('pair_codec_policy_ == "auto_exact"', bindings)
        self.assertIn("local_sparse_bytes + peer_sparse_bytes", bindings)
        self.assertIn("choose_exact_bucket_codec(", bindings)
        self.assertIn('"exact_sorted_sparse") == 0', bindings)
        self.assertNotIn('"sorted_sparse") == 0', bindings)
        self.assertIn('pair_gaussian_codec_ = used_sparse', bindings)
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        merge_start = trainer.index(
            "void VulkanGSTrainer::executeFTGSPPSparseGradientMerge("
        )
        merge = trainer[
            merge_start : trainer.index(
                "void VulkanGSTrainer::executeFTGSPPRelocation(",
                merge_start,
            )
        ]
        self.assertIn("scatter pair-sparse sources in rank order", merge)
        self.assertIn("if (local_rank == 0)", merge)
        self.assertNotIn("pipeline_ftgspp_sparse_prepare", merge)

    def test_non_logging_steps_gate_every_native_loss_scalar_path(self):
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        shared_start = trainer.index(
            "void VulkanGSTrainer::executeComputeSSIMGradient("
        )
        shared_end = trainer.index(
            "void VulkanGSTrainer::executeFTGSPPMaterialize(",
            shared_start,
        )
        shared = trainer[shared_start:shared_end]
        self.assertIn(
            "if (collect_loss_scalar && deviceInfo.hasFloat32AtomicAdd)",
            shared,
        )
        self.assertIn("clearDeviceBuffer(buffers.loss_accum, 6)", shared)
        self.assertIn("pipeline_ftgspp_loss_reduce", shared)
        self.assertIn("pending_collect_loss_scalar_ = collect_loss_scalar", bindings)
        self.assertIn("collect_loss=should_log", runner)

    def test_raster_backward_tuning_is_keyed_and_freezes_per_bucket(self):
        header = (ROOT / "native" / "src" / "render" / "gs_renderer.h").read_text(
            encoding="utf-8"
        )
        source = (ROOT / "native" / "src" / "render" / "gs_renderer.cpp").read_text(
            encoding="utf-8"
        )
        for field in (
            "image_width", "image_height", "tile_density_bucket", "active_sh",
        ):
            self.assertIn(field, header)
        self.assertIn("rasterizeBackwardTuningCache", header)
        self.assertIn("density_bucket", source)
        self.assertIn("raster_pair_count_for_density", source)
        self.assertIn("telemetry_.exact_tile_count", source)
        backward_start = source.index(
            "void VulkanGSRenderer::executeRasterizeBackward"
        )
        next_fn = source.find("\nvoid VulkanGSRenderer::", backward_start + 1)
        raster_bwd_body = source[
            backward_start : next_fn if next_fn != -1 else backward_start + 4000
        ]
        self.assertIn("raster_pair_count_for_density", raster_bwd_body)
        self.assertNotIn("(buffers.num_indices + tile_count", raster_bwd_body)
        self.assertNotIn("tile_pair_capacity", raster_bwd_body)
        self.assertIn("state.warmup_updates >= 16", source)
        self.assertIn("state.scheduler.fully_initialized()", source)
        self.assertIn("state.scheduler.best_mean()", source)

    def test_tile_pair_capacity_bootstraps_tighter_than_splat_times_eight(self):
        header = (ROOT / "native" / "src" / "render" / "tile_work_control.h").read_text(
            encoding="utf-8"
        )
        renderer = (ROOT / "native" / "src" / "render" / "gs_renderer.cpp").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("bootstrapTilePairCapacity", header)
        self.assertIn("growTilePairCapacity", header)
        self.assertIn("num_splats * 6", header)
        self.assertNotIn("num_splats * 8", renderer)
        self.assertIn("ftgspp::bootstrapTilePairCapacity", renderer)
        self.assertIn("ftgspp::growTilePairCapacity", bindings)
        self.assertIn("tile_capacity_headroom", bindings)
        self.assertIn("collectLastHarvestedOverflow", bindings)
        self.assertIn("grow_eval_tile_capacity", bindings)
        self.assertIn("uniforms.step = 0", bindings)
        prepare = (ROOT / "native" / "slang" / "ftgspp_tile_work_prepare.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn("if (u.step_generation != 0u)", prepare)

    def test_stage_timestamps_default_off_keeps_raster_backward_probes(self):
        pipeline = (ROOT / "native" / "src" / "device" / "gs_pipeline.cpp").read_text(
            encoding="utf-8"
        )
        timer = (ROOT / "native" / "src" / "device" / "perf_timer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("FTGSPP_VK_STAGE_TIMERS", pipeline)
        self.assertIn("stage_timestamps_all_", pipeline)
        self.assertIn("isRasterBackwardProbe", timer)
        self.assertIn("stageTimestampsEnabled()", timer)

    def test_eval_copies_packed_rgb_without_extra_numpy_roundtrip(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        backend = (ROOT / "src" / "ftgspp" / "training" / "backend.py").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("copy_pixel_rgb", bindings)
        self.assertIn("begin_eval_val_view", bindings)
        self.assertIn("submit_eval_rgb_readback", bindings)
        self.assertIn("complete_eval_rgb", bindings)
        self.assertIn("complete_eval_metrics", bindings)
        self.assertIn("drain_eval", bindings)
        self.assertIn("submitComputeReadback", bindings)
        self.assertIn("harvestNextComputeSlot", bindings)
        self.assertIn("rerender_last_eval_view", bindings)
        rerender = bindings[
            bindings.index("void rerender_last_eval_view") : bindings.index(
                "py::array_t<float> complete_eval_rgb"
            )
        ]
        self.assertIn("submitComputeReadback", rerender)
        self.assertNotIn("DeviceGuard", rerender)
        self.assertIn("complete_eval_rgb", backend)
        self.assertIn("complete_eval_metrics", backend)
        self.assertIn("evaluate_holdout_view", backend)
        holdout = backend[
            backend.index("def evaluate_holdout_view(") : backend.index(
                "def render_flow_diagnostic("
            )
        ]
        self.assertIn("complete_eval_metrics", holdout)
        self.assertNotIn("ssim_workspace", holdout)
        self.assertNotIn("_psnr(", holdout)
        self.assertNotIn("_color_correction_enabled", holdout)
        self.assertIn("begin(int(view_index), False)", holdout)
        render_eval = backend[
            backend.index("def render_eval(") : backend.index("def render_components(")
        ]
        self.assertIn("begin(int(view_index), False)", render_eval)
        self.assertNotIn("_color_correction_enabled", render_eval)
        components = backend[
            backend.index("def render_components(") : backend.index(
                "def evaluate_holdout_view("
            )
        ]
        self.assertNotIn("_color_correction_enabled", components)
        self.assertIn("begin(int(view_index), bool(apply_color_correction))", components)
        evaluation = runner[
            runner.index("def _evaluate(") : runner.index("def _optimizer_update(")
        ]
        self.assertIn("evaluate_holdout_view", evaluation)
        self.assertIn(
            "want_diagnostics = foreground_protocol and bool(full_validation)",
            evaluation,
        )
        self.assertIn("ssim_workspace.ssim", evaluation)
        self.assertIn("class _EvalSsimWorkspace", runner)
        self.assertNotIn("full_validation_on_export", evaluation)
        self.assertLess(
            evaluation.index("want_diagnostics = foreground_protocol"),
            evaluation.index("foreground_metrics("),
        )
        ssim = (SLANG / "ssim.slang").read_text(encoding="utf-8")
        self.assertIn("float ssimC1;", ssim)
        self.assertIn("uniforms.ssimC1", ssim)
        buffer_cpp = (ROOT / "native" / "src" / "device" / "buffer.cpp").read_text(
            encoding="utf-8"
        )
        readback = buffer_cpp[
            buffer_cpp.index("submitComputeReadback(") : buffer_cpp.index(
                "completeBuffersFromDevicePackedRange("
            )
        ]
        self.assertIn("attachComputeTimeline", readback)
        packed = buffer_cpp[
            buffer_cpp.index("submitBuffersFromDevicePackedRange(") : buffer_cpp.index(
                "submitComputeReadback("
            )
        ]
        self.assertIn("attachComputeTimeline", packed)
        self.assertIn("HOST_READ", packed)

    def test_train_log_interval_is_public_config_not_hidden_environment(self):
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        defaults = (ROOT / "src" / "ftgspp" / "config" / "defaults.yaml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("FTGSPP_TRAIN_LOG_INTERVAL", runner)
        self.assertIn("self.config.train.log_interval", runner)
        self.assertIn("log_interval: 2000", defaults)

    def test_eval_cadence_uses_checkpoint_interval_not_a_second_env_switch(self):
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("FTGSPP_TRAIN_EVAL_INTERVAL", runner)
        self.assertIn("def snapshot_cadence(", runner)
        self.assertIn("save_checkpoints", runner)
        self.assertIn("export_best", runner)
        self.assertIn("holdout_is_better", runner)
        self.assertIn("export_selection.json", runner)
        defaults = (ROOT / "src" / "ftgspp" / "config" / "defaults.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("save_checkpoints: true", defaults)
        self.assertIn("export_best: true", defaults)
        self.assertNotIn("full_validation_on_export", defaults)
        self.assertIn("full_validation=True", runner)
        schema = (ROOT / "src" / "ftgspp" / "config" / "schema.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("full_validation_on_export", schema)

    def test_pair_bucket_calibration_uses_real_native_round_trips(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        distributed = (
            ROOT / "src" / "ftgspp" / "training" / "distributed.py"
        ).read_text(encoding="utf-8")
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("benchmark_pair_transfer_buckets(", bindings)
        self.assertIn("submitBuffersFromDevicePackedRange(", bindings)
        self.assertIn("submitBuffersToDevicePackedRange(", bindings)
        self.assertIn("wall_aggregator", bindings)
        self.assertIn("median_of(", bindings)
        self.assertIn("for value in (8, 16, 32, 64)", distributed)
        self.assertIn("slower_rank_wall_seconds", distributed)
        self.assertIn("collective_bucket_benchmark", distributed)
        self.assertIn("collective_bucket_bytes=group.collective_bucket_bytes", runner)
        self.assertIn('saved_collective.get("bucket_bytes", 0)', runner)

    def test_color_backward_does_not_apply_adam(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        color_start = cpp.index(
            "void VulkanGSTrainer::executeFTGSPPColorCorrectBackwardRawWH"
        )
        color = cpp[color_start : cpp.index("\n}\n", color_start) + 3]
        self.assertNotIn("pipeline_ftgspp_color_adam", color)

    def test_sync_pipeline_harvests_at_collective_and_narrows_eval_drain(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        buffer_cpp = (ROOT / "native" / "src" / "device" / "buffer.cpp").read_text(
            encoding="utf-8"
        )
        ring = (ROOT / "native" / "src" / "device" / "pipeline_ring.cpp").read_text(
            encoding="utf-8"
        )
        distributed = (
            ROOT / "src" / "ftgspp" / "training" / "distributed.py"
        ).read_text(encoding="utf-8")
        backward = bindings[
            bindings.index("void backward_dynamic_view(") : bindings.index(
                "void train_dynamic_view_from_loss"
            )
        ]
        self.assertNotIn("resolve_tile_overflow", backward)
        self.assertNotIn("harvestNextComputeSlot", backward)
        self.assertIn("record_pending_tile_step", backward)
        self.assertIn("setSubmissionMeta", backward)
        self.assertIn("harvest_sync_generation", bindings)
        self.assertIn("inject_sync_tile_overflow", bindings)
        self.assertIn("apply_sync_tile_capacity", bindings)
        self.assertIn("relocation_stats_generation", bindings)
        apply_capacity = bindings[
            bindings.index("void apply_sync_tile_capacity(") : bindings.index(
                "void inject_sync_tile_overflow("
            )
        ]
        self.assertIn("relocation_stats_generation", apply_capacity)
        self.assertIn("copyFromDeviceToDevice", apply_capacity)
        raw_backward = bindings[
            bindings.index("void backward_dynamic_view_from_loss_raw(")
            : bindings.index("void set_collect_geometry_grad_stats(")
        ]
        self.assertIn("relocation_stats_generation", raw_backward)
        drain = bindings[
            bindings.index("void drain_eval()") : bindings.index(
                "void drain_for_structure_change"
            )
        ]
        self.assertIn("waitAllComputeSlots", drain)
        self.assertNotIn("resolve_tile_overflow", drain)
        self.assertIn("attachComputeTimeline", buffer_cpp)
        self.assertIn("drainPackedTransfers", ring)
        self.assertIn("_complete_generation_with_overflow_replay", distributed)
        self.assertIn("sync_abort_retries", distributed)

    def test_parallel_eval_render_entry_points_release_the_gil(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        for name in (
            "render_dynamic_train_view", "render_dynamic_val_view",
            "render_flow_dynamic_train_view", "render_flow_dynamic_val_view",
        ):
            self.assertRegex(
                bindings,
                rf'\.def\("{name}"[\s\S]{{0,220}}'
                r'py::call_guard<py::gil_scoped_release>\(\)',
            )

    def test_distributed_transfers_release_gil_around_vulkan_waits(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        helper_start = bindings.index("py::array_t<To> buffer_to_array(")
        helper = bindings[
            helper_start : bindings.index("\n    }\n", helper_start) + 7
        ]
        self.assertIn("py::gil_scoped_release release;", helper)
        self.assertLess(
            helper.index("py::gil_scoped_release release;"),
            helper.index("trainer.copyFromDevice(buffer);"),
        )

    def test_named_gradient_abi_is_removed(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("py::dict gradient_state(", bindings)
        self.assertNotIn("void set_gradient_state(py::dict", bindings)
        self.assertNotIn('.def("gradient_state"', bindings)
        self.assertNotIn('.def("set_gradient_state"', bindings)

    def test_optimizer_relocation_statistics_keep_n_by_two_shape(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        optimizer_start = bindings.index("py::dict get_optimizer_state()")
        optimizer = bindings[
            optimizer_start : bindings.index("\n    }\n", optimizer_start) + 7
        ]
        self.assertRegex(
            optimizer,
            r'state\["relocation_stats"\]\s*=\s*buffer_to_array<float, float>\('
            r'\s*buffers\.relocation_stats,\s*2\s*\)',
        )
        relocation_start = bindings.index("py::array_t<float> relocation_stats_state()")
        relocation = bindings[
            relocation_start : bindings.index("\n    }\n", relocation_start) + 7
        ]
        self.assertIn("buffer_to_array<float, float>(buffers.relocation_stats, 2)", relocation)
        self.assertIn('"relocation statistics must have shape (N, 2)"', bindings)

    def test_perf_timer_markers_are_thread_local_and_summary_is_global(self):
        source = (ROOT / "native" / "src" / "device" / "perf_timer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("struct _TimerObject {", source)
        self.assertNotIn("static struct _TimerObject {", source)
        self.assertIn("static std::mutex stagesMutex;", source)
        self.assertIn("static _TimerObject stages[TrainStage::END];", source)
        self.assertNotIn("thread_local _TimerObject stages", source)
        self.assertGreaterEqual(
            source.count("std::lock_guard<std::mutex> lock(stagesMutex);"), 4
        )
        for name in ("marks", "pushedMarks", "hostHold", "hostStartTime", "hostTimeDelta"):
            self.assertRegex(source, rf"static thread_local [^;]*\b{name}\b")

    def test_gated_occupancy_follows_phi_not_raw_marginal(self):
        dynamic = (SLANG / "ftgspp_dynamic.slang").read_text(encoding="utf-8")
        self.assertIn(
            "float temporal_opacity = gate + (1.0 - gate) * marginal;",
            dynamic,
        )
        self.assertIn(
            "if (temporal_opacity <= marginal_threshold)",
            dynamic,
        )
        self.assertNotIn("if (marginal <= marginal_threshold)", dynamic)
        for name in (
            "ftgspp_fused_backward_optimizer.slang",
            "ftgspp_fused_dynamic_backward_adam.slang",
        ):
            shader = (SLANG / name).read_text(encoding="utf-8")
            self.assertIn(
                "const bool active = mg.temporal_opacity > uniforms.marginal_threshold;",
                shader,
                name,
            )
            self.assertNotIn(
                "const bool active = marginal > uniforms.marginal_threshold;",
                shader,
                name,
            )

    def test_hard_separation_phase_is_available(self):
        self.assertTrue((SLANG / "ftgspp_hard_separation.slang").exists())
        compiler = (ROOT / "native" / "compile_shaders.py").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        backend = (ROOT / "src" / "ftgspp" / "training" / "backend.py").read_text(
            encoding="utf-8"
        )
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ftgspp_hard_separation", compiler)
        self.assertIn("begin_hard_separation", bindings)
        self.assertIn("resume_hard_separation", bindings)
        self.assertIn("hard_separation_gate_state", bindings)
        self.assertIn("begin_hard_separation", backend)
        self.assertIn("hard_separation_digest", backend)
        self.assertIn("hard_phase_seed", runner)
        self.assertIn('phase="hard_separation"', runner)

    def test_color_adam_is_disabled_during_hard_separation(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        adam_start = cpp.index("void VulkanGSTrainer::executeFTGSPPColorAdam(")
        adam = cpp[adam_start : cpp.index("\n}\n", adam_start) + 3]
        self.assertIn("hard_separation", adam)
        self.assertIn(
            "config.color_reg,\n        1u | (collect_loss_scalar ? 2u : 0u),",
            adam,
        )
        self.assertIn("pipeline_ftgspp_color_adam", adam)
        self.assertNotIn("clearDeviceBuffer(buffers.color_grad, param_count);", adam)

        shader = (ROOT / "native" / "slang" / "ftgspp_color_adam.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "color_grad[i] + 2.0f * u.reg * color_params[i]", shader
        )
        self.assertIn("if ((u.update_parameters & 1u) == 0u)", shader)
        self.assertLess(
            shader.index("loss_accum.InterlockedAddF32"),
            shader.index("if ((u.update_parameters & 1u) == 0u)"),
        )

        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        active_start = bindings.index("bool color_correction_active(int step) const")
        active = bindings[
            active_start : bindings.index("\n    }\n", active_start) + 7
        ]
        self.assertIn("step > config.color_correction_start", active)
        self.assertIn("config.hard_separation", active)

    def test_eval_render_can_apply_color_correction_when_requested(self):
        """Eval has an explicit CC switch; default remains raw for compatibility."""
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        train_start = bindings.index("void render_dynamic_train_view(")
        train = bindings[
            train_start : bindings.index("void set_color_correction_params(", train_start)
        ]
        self.assertIn("apply_color_correction", train)
        self.assertIn("executeFTGSPPColorCorrectForwardWH", train)

        val_start = bindings.index("void render_dynamic_val_view(")
        val = bindings[val_start : bindings.index("\n    }\n", val_start) + 7]
        self.assertIn("apply_color_correction", val)
        self.assertIn("executeFTGSPPColorCorrectForwardWH", val)

    def test_color_correction_orders_transfer_and_ssim_producers(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        forward_start = cpp.index(
            "void VulkanGSTrainer::executeFTGSPPColorCorrectForwardWH("
        )
        forward = cpp[forward_start : cpp.index("\n}\n", forward_start) + 3]
        self.assertIn("resizeDeviceBuffer(buffers.color_raw_rgb", forward)
        self.assertNotIn("copyFromDeviceToDevice(", forward)
        self.assertIn("buffers.color_raw_rgb.deviceBuffer", forward)
        self.assertIn(
            "buffers.pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE",
            forward,
        )

        backward_start = cpp.index(
            "void VulkanGSTrainer::executeFTGSPPColorCorrectBackwardRawWH("
        )
        backward = cpp[
            backward_start : cpp.index("\n}\n", backward_start) + 3
        ]
        self.assertIn(
            "v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE", backward
        )
        self.assertIn("color_grad.deviceBuffer, TRANSFER_WRITE", backward)
        self.assertIn("accumulate_parameter_grads ? 1u : 2u", backward)
        self.assertIn("COMPUTE_SHADER_READ_WRITE", backward)

        shader = (
            ROOT / "native" / "slang" / "ftgspp_color_correct_backward.slang"
        ).read_text(encoding="utf-8")
        self.assertIn("if (u.enabled == 1u)", shader)

    def test_color_state_is_deterministically_preallocated_per_replica(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        start = cpp.index("void VulkanGSTrainer::ensureFTGSPPColorState(")
        helper = cpp[start : cpp.index("\n}\n", start) + 3]
        for name in ("color_params", "color_m", "color_v"):
            self.assertIn(f"buffers.{name}.deviceSize() == param_count", helper)
            self.assertIn(f"buffers.{name}.assign(param_count, 0.0f)", helper)
            self.assertIn(f"copyToDevice(buffers.{name})", helper)
        self.assertLess(helper.index("return;"), helper.index("assign(param_count"))

        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        configure_start = bindings.index("void configure_ftgspp(py::dict cfg)")
        configure = bindings[
            configure_start : bindings.index("\n    }\n", configure_start) + 7
        ]
        self.assertIn(
            "trainer.ensureFTGSPPColorState(config, buffers);", configure
        )

    def test_relocation_cdf_uses_parallel_float_scan(self):
        cpp = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        renderer = (ROOT / "native" / "src" / "render" / "gs_renderer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("executeFTGSPPRelocationPlanNative(", cpp)
        self.assertIn("executeFloatInclusiveScan(", cpp)
        self.assertIn("void VulkanGSRenderer::executeFloatInclusiveScan(", renderer)
        self.assertNotIn(
            "executeCompute({{1,1}}, &u, sizeof(u), pipeline_ftgspp_relocation_cdf",
            cpp,
        )
        self.assertIn("pipeline_ftgspp_relocation_cdf_normalize", cpp)
        shader = (SLANG / "ftgspp_relocation_plan.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn("donor_cdf[u.num_splats - 1u]", shader)
        self.assertNotIn("donor_cdf[i] /= cumulative", shader)

    def test_multi_rank_relocation_skips_redundant_host_roundtrips(self):
        distributed = (
            ROOT / "src" / "ftgspp" / "training" / "distributed.py"
        ).read_text(encoding="utf-8")
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        start = distributed.index(
            "def relocate(\n        self, step: int, mode: str, include_gate: bool, seed: int"
        )
        body = distributed[start : distributed.index("\n    def checkpoint_optimizer_state", start)]
        self.assertIn("self._sum_relocation_statistics()", body)
        self.assertNotIn("self._global_relocation_statistics()", body)
        self.assertNotIn("drain_for_structure_change", body)
        self.assertNotIn("set_relocation_stats_state(zero_statistics)", body)
        self.assertIn("backend.apply_planned_relocation(str(mode))", body)
        self.assertIn("clear_relocation_stats_state()", body)
        self.assertIn("void apply_planned_relocation(", bindings)
        self.assertIn("void clear_relocation_stats_state()", bindings)

    def test_relocation_degeneracy_replaces_the_whole_donor_distribution(self):
        shader = (SLANG / "ftgspp_relocation_plan.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn("relocation_uniform_fallback", shader)
        self.assertIn("relocation_donor_score", shader)
        self.assertIn("if (uniform_fallback)", shader)
        for component in ("sum.x", "sum.y", "sum.z"):
            self.assertIn(f"!isfinite({component})", shader)
            self.assertIn(f"{component} <= 0.0f", shader)
        self.assertIn(
            "status[3] = uniform_fallback && alive_count != 0u ? 1u : 0u;",
            shader,
        )
        self.assertNotIn(
            "sum.x > 0.0f ? raw.x/sum.x : 1.0f/float(u.num_splats)",
            shader,
        )

    def test_relocation_equation_uses_exact_ftgspp_float32_binomials(self):
        table = (SLANG / "ftgspp_relocation_binoms.slang").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"FTGSPP_RELOCATION_BINOM_BITS\[676\]\s*=\s*\{(.*?)\};",
            table,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        actual = [
            int(value, 16)
            for value in re.findall(r"0x([0-9a-fA-F]{8})u", match.group(1))
        ]
        expected = []
        for n in range(51):
            for k in range(n // 2 + 1):
                expected.append(
                    struct.unpack("<I", struct.pack("<f", math.comb(n, k)))[0]
                )
        self.assertEqual(len(actual), 676)
        self.assertEqual(actual, expected)
        self.assertIn("FTGSPP_RELOCATION_MAX_RATIO = 51u", table)

        prepare = (SLANG / "ftgspp_relocation_prepare.slang").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "min(copies + 1, int(FTGSPP_RELOCATION_MAX_RATIO))", prepare
        )
        self.assertIn("ftgspp_relocation_binom(uint(i - 1), uint(k))", prepare)
        self.assertIn("(k & 1) != 0 ? -1.0f : 1.0f", prepare)
        self.assertIn("1.0f - FLOAT32_EPSILON", prepare)
        self.assertNotIn("log_factorial", prepare)
        self.assertNotIn("cos(", prepare)
        self.assertNotIn("max(denom", prepare)
        self.assertNotIn("log(max(coeff", prepare)
        self.assertIn("float3 new_log_scale = log(coeff * exp(raw.xyz));", prepare)
        for value in ("denom", "coeff", "new_log_scale.x"):
            self.assertIn(f"!isfinite({value})", prepare)

        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        start = bindings.index("void relocate_dynamic(")
        body = bindings[start : bindings.index("\n    }\n", start) + 7]
        self.assertIn(
            "trainer.copyFromDevice(buffers.base_log_scales_opacity);", body
        )
        self.assertIn("check_opacity_roles", body)
        self.assertIn("if (check_opacity_roles)", body)
        self.assertIn("require_dead_destinations", body)
        self.assertIn("require_alive_donors", body)
        self.assertIn("relocation destination is not dead", body)
        self.assertIn("relocation donor is dead/non-finite", body)
        self.assertIn("opacity_tolerance", body)
        self.assertIn(
            "destination_opacity > config.min_opacity + opacity_tolerance", body
        )
        self.assertIn(
            "donor_opacity <= config.min_opacity - opacity_tolerance", body
        )
        self.assertIn("Rank 0 owns the mapping", body)
        self.assertIn("relocation donor counts do not match", body)
        self.assertIn("relocation_max_ratio = 51", body)

        compiler = (ROOT / "native" / "compile_shaders.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"ftgspp_relocation_binoms.slang"', compiler)

    def test_single_gpu_view_backward_uses_one_pre_step_parameter_point(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        low_level_start = bindings.index("void train_dynamic_view(")
        low_level = bindings[
            low_level_start : bindings.index(
                "py::array_t<float> copy_pixel_rgb", low_level_start
            )
        ]
        self.assertIn("train_dynamic_view_to_loss(", low_level)
        self.assertIn("train_dynamic_view_from_loss(", low_level)
        self.assertNotIn("executeFTGSPPBackward(", low_level)
        self.assertNotIn("executeFTGSPPExplicitAdam", low_level)
        self.assertNotIn("executeFTGSPPBackwardOptimizerStep", low_level)

        raw_start = bindings.index(
            "void backward_dynamic_view_from_loss_raw(size_t view_idx, int step)"
        )
        raw = bindings[
            raw_start : bindings.index("void optimizer_step(int step)", raw_start)
        ]
        self.assertLess(
            raw.index("executeFTGSPPColorCorrectBackwardRawWH"),
            raw.index("rasterize_backward()"),
        )
        self.assertLess(
            raw.index("rasterize_backward()"),
            raw.index("executeFTGSPPBackward("),
        )
        self.assertIn("cannot overwrite pending", raw)

        optimizer_start = bindings.index("void optimizer_step(int step)")
        optimizer = bindings[
            optimizer_start : bindings.index("void backward_dynamic_view(", optimizer_start)
        ]
        self.assertLess(
            optimizer.index("executeFTGSPPExplicitAdam"),
            optimizer.index("executeFTGSPPColorAdam"),
        )

        from_loss_start = bindings.index("void train_dynamic_view_from_loss(")
        from_loss = bindings[
            from_loss_start : bindings.index("void train_dynamic_view(", from_loss_start)
        ]
        self.assertLess(
            from_loss.index("executeFTGSPPColorCorrectBackwardRawWH"),
            from_loss.index("rasterize_backward()"),
        )
        self.assertLess(
            from_loss.index("rasterize_backward()"),
            from_loss.index("executeFTGSPPFusedDynamicBackwardAdam("),
        )
        self.assertLess(
            from_loss.index("executeFTGSPPFusedDynamicBackwardAdam("),
            from_loss.index("executeFTGSPPColorAdam("),
        )
        self.assertNotIn("backward_dynamic_view_from_loss_raw", from_loss)
        self.assertNotIn("optimizer_step(step)", from_loss)
        self.assertNotIn("executeFTGSPPBackwardOptimizerStep", from_loss)
        self.assertNotIn("gradient_ready = false", from_loss)

        to_loss_start = bindings.index("void train_dynamic_view_to_loss(")
        to_loss = bindings[to_loss_start:raw_start]
        self.assertIn("cannot overwrite pending gradients", to_loss)

    def test_cleanup_waits_before_destroying_device_children(self):
        source = (ROOT / "native" / "src" / "device" / "gs_pipeline.cpp").read_text(encoding="utf-8")
        start = source.index("void VulkanGSPipeline::cleanup()")
        body = source[start : source.index("\n}\n", start) + 3]
        self.assertLess(body.index("vkDeviceWaitIdle(device)"), body.index("destroyStager"))
        self.assertLess(body.index("vkDeviceWaitIdle(device)"), body.index("destroyComputePipeline"))
        self.assertLess(body.index("destroyComputePipeline"), body.index("vkDestroyDevice"))
        self.assertIn("destroyStager(stager)", body)
        self.assertIn("destroyStager(readback_stager)", body)
        self.assertIn("destroyStager(upload_stager)", body)

    def test_zero_dispatch_returns_before_descriptor_update(self):
        source = (ROOT / "native" / "src" / "device" / "gs_pipeline.cpp").read_text(encoding="utf-8")
        start = source.index("void VulkanGSPipeline::executeCompute(")
        body = source[start : source.index("\n}\n", start) + 3]
        self.assertLess(
            body.index("if (dim.first == 0)"),
            body.index("bindComputePipeline("),
        )
        self.assertNotIn("updateComputeDescriptorSet(pipeline, buffers)", body)
        bind_start = source.index("void VulkanGSPipeline::bindComputePipeline(")
        bind_body = source[bind_start : source.index("\n}\n", bind_start) + 3]
        self.assertIn("updateComputeDescriptorSet(pipeline, buffers)", bind_body)

    def test_buffer_ownership_is_move_only(self):
        header = (ROOT / "native" / "src" / "device" / "buffer.h").read_text(encoding="utf-8")
        self.assertRegex(header, r"Buffer\(const Buffer\s*&other\)\s*=\s*delete")
        self.assertRegex(header, r"Buffer& operator=\(const Buffer\s*&other\)\s*=\s*delete")
        self.assertIn("Buffer(Buffer&&", header)
        self.assertIn("generation == other.generation", header)

    def test_raw_gradient_buffers_are_reported_and_destroyed(self):
        accounting = (ROOT / "native" / "src" / "device" / "buffer.cpp").read_text(
            encoding="utf-8"
        )
        lifecycle = (ROOT / "native" / "src" / "device" / "gs_pipeline.cpp").read_text(
            encoding="utf-8"
        )
        for name in (
            "grad_base_xyz_ws",
            "grad_base_log_scales_opacity",
            "grad_times_durations",
            "grad_velocities_gates",
            "grad_sh_coeffs",
            "grad_rotations",
        ):
            self.assertIn(f"_({name})", accounting)
            self.assertIn(f"_({name})", lifecycle)

    def test_python_exposes_only_pair_host_transport(self):
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        distributed = (
            ROOT / "src" / "ftgspp" / "training" / "distributed.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("py::dict gradient_state(", bindings)
        self.assertNotIn("void set_gradient_state(py::dict", bindings)
        self.assertIn(
            'requested_transport not in {"auto", "pair_host"}',
            distributed,
        )
        self.assertNotIn("def _train_step_device_group", distributed)
        self.assertNotIn("def _ensure_packed_layout", distributed)
        self.assertIn("pair_exchange_gpu_merge", distributed)
        self.assertIn("exchange_pair_gradient", bindings)
        self.assertIn("optimizer_step_pair_exchange_gpu_merge", bindings)
        self.assertNotIn("exchange_host_mailbox_peer_gradient", bindings)
        self.assertNotIn("optimizer_step_host_mailbox", bindings)

    def test_internal_pair_capability_uses_async_borrowed_readback_ring(self):
        pipeline_h = (ROOT / "native" / "src" / "device" / "gs_pipeline.h").read_text(
            encoding="utf-8"
        )
        pipeline_cpp = (ROOT / "native" / "src" / "device" / "buffer.cpp").read_text(
            encoding="utf-8"
        )
        owner_cpp = (
            ROOT / "native" / "src" / "collective" / "host_mailbox_owner.cpp"
        ).read_text(encoding="utf-8")
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("packed_transfer_ring_depth = 8", pipeline_h)
        self.assertIn("StagingPurpose::CachedUpload", pipeline_cpp)
        self.assertIn("kMinSparsePairBucketBytes", bindings)
        self.assertIn("submitBuffersFromDevicePackedRange", pipeline_cpp)
        self.assertIn("submitBuffersToDevicePackedRange", pipeline_cpp)
        self.assertIn("completeBuffersToDevicePackedRange", bindings)
        self.assertIn("ftgspp-host-mailbox-v3", owner_cpp)
        self.assertIn('{"host_storage_bytes", static_cast<uint64_t>(0)}', owner_cpp)
        self.assertIn('{"borrowed_storage", true}', owner_cpp)
        self.assertIn('result["async_pipeline"] = true', bindings)

    def test_packed_transport_batches_each_direction_in_one_command_submission(self):
        source = (ROOT / "native" / "src" / "device" / "buffer.cpp").read_text(
            encoding="utf-8"
        )
        read_start = source.index("void VulkanGSPipeline::copyBuffersFromDevicePacked(")
        read_body = source[
            read_start : source.index("void VulkanGSPipeline::copyBuffersToDevicePacked(", read_start)
        ]
        write_start = source.index("void VulkanGSPipeline::copyBuffersToDevicePacked(")
        write_body = source[
            write_start : source.index(
                "namespace {\nstruct PackedRangeIntersection",
                write_start,
            )
        ]
        self.assertEqual(read_body.count("DEVICE_GUARD;"), 1)
        self.assertEqual(write_body.count("DEVICE_GUARD;"), 1)
        self.assertEqual(read_body.count("HOST_GUARD;"), 1)
        self.assertEqual(write_body.count("HOST_GUARD;"), 1)
        self.assertEqual(read_body.count("memcpy("), 1)
        self.assertEqual(write_body.count("memcpy("), 1)
        self.assertIn(
            "allocStagingBufferFor(readback_stager, packedSize, StagingPurpose::Readback);",
            read_body,
        )
        self.assertIn(
            "allocStagingBufferFor(upload_stager, packedSize, StagingPurpose::Upload);",
            write_body,
        )
        self.assertIn("ensureStagerMapped(readback_stager)", read_body)
        self.assertIn("ensureStagerMapped(upload_stager)", write_body)

    def test_device_identity_comes_from_device_info_not_group_probe(self):
        source = (ROOT / "native" / "src" / "device" / "gs_pipeline.cpp").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("get_device_group_info", source)
        self.assertNotIn("vkEnumeratePhysicalDeviceGroups", source)
        self.assertNotIn("vkGetDeviceGroupPeerMemoryFeatures(", source)
        start = source.index("VulkanGSPipeline::get_device_info() const")
        body = source[start : source.index("void VulkanGSPipeline::createDevice()", start)]
        self.assertIn('result["device_uuid"]', body)
        self.assertIn("VkPhysicalDeviceIDProperties", body)
        self.assertNotIn('result["peer_memory_queried"]', body)
        self.assertNotIn('result["all_reduce_ready"]', body)

    def test_native_device_provenance_reports_driver_uuid_and_features(self):
        pipeline = (ROOT / "native" / "src" / "device" / "gs_pipeline.cpp").read_text(
            encoding="utf-8"
        )
        start = pipeline.index("VulkanGSPipeline::get_device_info() const")
        body = pipeline[
            start : pipeline.index("void VulkanGSPipeline::createDevice()", start)
        ]
        for key in (
            "vendor_id", "device_id_pci", "driver_version", "api_version",
            "subgroup_size", "has_int16", "has_int64",
            "has_float32_atomic_add", "device_uuid",
        ):
            self.assertIn(f'result["{key}"]', body)
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('.def("device_info"', bindings)
        self.assertIn('.def("native_abi"', bindings)
        self.assertNotIn('.def("device_group_capability"', bindings)
        backend = (ROOT / "src" / "ftgspp" / "training" / "backend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def device_metadata(self)", backend)
        self.assertIn('info.get("device_uuid"', backend)
        self.assertNotIn("device_group_capability", backend)
        self.assertNotIn('"physical_group_signature"', backend)
        self.assertIn("def reported_native_abi(self)", backend)

    def test_checkpoint_readback_does_not_duplicate_owned_numpy_arrays(self):
        backend = (ROOT / "src" / "ftgspp" / "training" / "backend.py").read_text(
            encoding="utf-8"
        )
        model_start = backend.index("def model(self)")
        body = backend[
            model_start : backend.index("def optimizer_state(self)", model_start)
        ]
        self.assertIn("np.ascontiguousarray(value, dtype=np.float32)", body)
        self.assertNotIn("np.array(value, copy=True)", body)

        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        optimizer_start = bindings.index("py::dict get_optimizer_state()")
        optimizer = bindings[
            optimizer_start : bindings.index("void set_optimizer_state(", optimizer_start)
        ]
        model_start = bindings.index("py::dict get_dynamic_gaussians()")
        model = bindings[
            model_start : bindings.index("size_t num_train()", model_start)
        ]
        self.assertIn("release_optimizer_host_copies();", optimizer)
        self.assertIn("release_model_host_copies();", model)
        self.assertIn('.def("release_model_upload_host_copies"', bindings)
        self.assertIn("release_model_upload_host_copies", backend)

    def test_resident_image_loading_is_removed(self):
        backend = (ROOT / "src" / "ftgspp" / "training" / "backend.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("resident", backend.lower())
        self.assertNotIn("def _upload_dataset_shared_resident(", backend)
        self.assertNotIn("def prefill_arrays(", backend)
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("resident", runner.lower())
        self.assertNotIn(
            "decoded RGB/native image residency is duplicated on every rank",
            runner,
        )

    def test_native_runner_has_no_host_relocation_planner(self):
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("def _relocation_plan", runner)
        self.assertIn("backend.relocate_native(", runner)

    def test_train_logs_include_per_rank_view_audit(self):
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            runner.count('"rank_view_audit": _rank_view_audit('), 2
        )
        helper = runner[
            runner.index("def _rank_view_audit(") :
            runner.index("\n\n@dataclass", runner.index("def _rank_view_audit("))
        ]
        for field in ('"rank"', '"view"', '"frame"', '"camera"', '"timestamp"'):
            self.assertIn(field, helper)

    def test_lpips_evaluation_is_unweighted_scalar_only(self):
        plugin = (
            ROOT / "src" / "ftgspp" / "optional" / "lpips.py"
        ).read_text(encoding="utf-8")
        self.assertIn("exclusively during evaluation", plugin)
        self.assertIn("evaluate_lpips_vgg_unnormalized", plugin)
        self.assertIn('reduction="none"', plugin)
        self.assertNotIn("LpipsPlugin", plugin)
        self.assertNotIn("require_gradient", plugin)

        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        evaluation = runner[
            runner.index("def _evaluate(") : runner.index("def _optimizer_update(")
        ]
        self.assertIn("evaluate_lpips_vgg_unnormalized", evaluation)
        self.assertIn("foreground_roi_lpips", evaluation)
        self.assertIn("ssim_workspace.ssim(", evaluation)
        self.assertIn("want_diagnostics", evaluation)

        backend = (
            ROOT / "src" / "ftgspp" / "training" / "backend.py"
        ).read_text(encoding="utf-8")
        training = backend[
            backend.index("def train_view(") : backend.index("def backward_view(")
        ]
        self.assertNotIn("lpips", training.lower())
        self.assertNotIn("accumulate_pixel_state_grad", backend)

    def test_resume_finalization_preserves_hard_separation_phase(self):
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("main_steps_executed", runner)
        self.assertIn('resume_phase == "main_all_dead"', runner)
        self.assertIn("is_final_main_step = step == self.config.train.iterations - 1", runner)
        self.assertIn('"main_complete"', runner)
        self.assertIn("if is_final_main_step", runner)
        self.assertIn('phase="hard_separation"', runner)
        self.assertIn("hard_separation_complete", runner)

    def test_runner_samples_main_and_hard_phase_views(self):
        runner = (ROOT / "src" / "ftgspp" / "training" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('phase="main"', runner)
        self.assertIn("seed=main_phase_seed", runner)
        self.assertIn("_sample_batch", runner)
        self.assertIn("hard_phase_seed", runner)
        self.assertIn("_sample_hard_batch", runner)
        self.assertIn('phase="hard_separation"', runner)

    def test_ftgspp_float_atomic_capability_is_hard_gated(self):
        header = (ROOT / "native" / "src" / "training" / "gs_trainer.h").read_text(
            encoding="utf-8"
        )
        trainer = (ROOT / "native" / "src" / "training" / "gs_trainer.cpp").read_text(
            encoding="utf-8"
        )
        bindings = (ROOT / "native" / "src" / "bindings" / "python_bindings.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("void requireFTGSPPFloat32AtomicAdd() const;", header)

        initialize = trainer[
            trainer.index("void VulkanGSTrainer::initialize(") :
            trainer.index("void VulkanGSTrainer::requireFTGSPPFloat32AtomicAdd() const")
        ]
        guarded_start = initialize.index("if (deviceInfo.hasFloat32AtomicAdd) {")
        guarded = initialize[guarded_start:]
        for pipeline in (
            "pipeline_ftgspp_loss_reduce",
            "pipeline_ftgspp_eval_metrics",
            "pipeline_ftgspp_fused_backward_optimizer",
            "pipeline_ftgspp_relocation_score",
            "pipeline_ftgspp_color_correct_backward",
            "pipeline_ftgspp_color_adam",
        ):
            self.assertIn(pipeline, guarded)

        requirement = trainer[
            trainer.index("void VulkanGSTrainer::requireFTGSPPFloat32AtomicAdd() const") :
            trainer.index("void VulkanGSTrainer::camera_to_uniforms(")
        ]
        self.assertIn("shaderBufferFloat32AtomicAdd", requirement)
        self.assertIn("alpha-blend CAS fallback", requirement)
        self.assertIn("Forward-only", requirement)

        # Defensive native entry points must never dispatch one of the
        # deliberately uncreated pipelines on a soft-capability device.
        for name in (
            "executeFTGSPPFusedDynamicBackwardAdam",
            "executeFTGSPPBackward",
            "executeFTGSPPRelocationPlanNative",
            "executeFTGSPPColorCorrectBackwardRawWH",
            "executeFTGSPPColorAdam",
            "executeEvalMetrics",
        ):
            start = trainer.index(f"void VulkanGSTrainer::{name}(")
            self.assertIn(
                "requireFTGSPPFloat32AtomicAdd();",
                trainer[start : start + 1200],
                name,
            )

        # Forward-only rendering can create the shared SSIM pipelines without
        # creating FTGSPP's float-atomic scalar-reduction pipeline.
        shared_ssim = trainer[
            trainer.index("void VulkanGSTrainer::executeComputeSSIMGradient(") :
            trainer.index("void VulkanGSTrainer::executeEvalMetrics(")
        ]
        self.assertIn(
            "if (collect_loss_scalar && deviceInfo.hasFloat32AtomicAdd) {",
            shared_ssim,
        )
        self.assertIn("pipeline_ftgspp_loss_reduce", shared_ssim)
        self.assertNotIn("requireFTGSPPFloat32AtomicAdd();", shared_ssim)
        self.assertIn("0.01f * 0.01f", shared_ssim)

        eval_metrics = trainer[
            trainer.index("void VulkanGSTrainer::executeEvalMetrics(") :
            trainer.index("void VulkanGSTrainer::executeFTGSPPMaterialize(")
        ]
        self.assertIn("requireFTGSPPFloat32AtomicAdd();", eval_metrics)
        self.assertIn("pipeline_ftgspp_eval_metrics", eval_metrics)
        self.assertIn("0.02f * 0.02f", eval_metrics)
        self.assertIn("eval_validity", eval_metrics)

        to_loss_start = bindings.index("void train_dynamic_view_to_loss(")
        to_loss = bindings[
            to_loss_start : bindings.index("void backward_dynamic_view_from_loss_raw(", to_loss_start)
        ]
        self.assertLess(
            to_loss.index("requireFTGSPPFloat32AtomicAdd();"),
            to_loss.index("dynamic_forward(timestamp)"),
        )
        for name in (
            "train_dynamic_view_from_loss",
            "backward_dynamic_view_from_loss_raw",
            "optimizer_step",
        ):
            definition = re.search(
                rf"^    \S[^\n]*\s{re.escape(name)}\(",
                bindings,
                flags=re.MULTILINE,
            )
            self.assertIsNotNone(definition, name)
            start = definition.start()
            self.assertIn(
                "requireFTGSPPFloat32AtomicAdd();",
                bindings[start : start + 900],
                name,
            )

        for shader_name in (
            "ftgspp_loss_reduce.slang",
            "ftgspp_eval_metrics.slang",
            "ftgspp_fused_backward_optimizer.slang",
            "ftgspp_relocation_plan.slang",
            "ftgspp_color_correct_backward.slang",
            "ftgspp_color_adam.slang",
        ):
            shader = (SLANG / shader_name).read_text(encoding="utf-8")
            self.assertIn("InterlockedAddF32", shader, shader_name)

if __name__ == "__main__":
    unittest.main()
