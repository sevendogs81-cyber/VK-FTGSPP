#!/usr/bin/env python3
"""
Shader compiler for the FTGSPP Vulkan runtime.
"""

import os
import sys
import hashlib
import subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
import argparse
import shutil
import json


@dataclass
class CompileConfig:
    """Configuration for shader compilation"""
    slang_src_path: Path = Path("./native/slang")
    shader_dst_path: Path = Path("./build/native/shaders")
    generated_dst_path: Path = Path("./build/native/shaders/generated")
    # slangc_compile_args: str = "-stage compute -O3 -fp-mode fast -line-directive-mode none"
    slangc_compile_args: str = "-stage compute -O -fp-mode fast -line-directive-mode none"
    glslc_compile_args: str = "-O --target-spv=spv1.5 --target-env=vulkan1.2"
    slangc_path: Optional[Path] = None
    glslc_path: Optional[Path] = None
    max_workers: Optional[int] = None


@dataclass
class ShaderJob:
    """Represents a single shader compilation job"""
    name: str
    defines: Dict[str, int]


class ShaderCompiler:
    def __init__(self, config: CompileConfig):
        self.config = config
        self.glsl_via_slangc = False
        self.slangc = self._find_slangc()
        self.glslc = self._find_glslc()
        
        self.config.shader_dst_path.mkdir(exist_ok=True)
        self.config.generated_dst_path.mkdir(exist_ok=True)

    def _find_slangc(self) -> Path:
        """Find slangc executable"""
        if self.config.slangc_path and self.config.slangc_path.exists():
            return self.config.slangc_path
        
        slangc = shutil.which("slangc")
        if slangc:
            return Path(slangc)
        
        raise RuntimeError("slangc not found. Please install Slang or specify path with --slangc")

    def _find_glslc(self) -> Path:
        """Find glslc, or fall back to slangc which can ingest GLSL compute."""
        if self.config.glslc_path and self.config.glslc_path.exists():
            name = self.config.glslc_path.name.lower()
            self.glsl_via_slangc = "slangc" in name
            return self.config.glslc_path
        
        glslc = shutil.which("glslc")
        if glslc:
            return Path(glslc)
        
        slangc = shutil.which("slangc")
        if slangc:
            self.glsl_via_slangc = True
            return Path(slangc)
        
        raise RuntimeError(
            "glslc not found and slangc fallback is unavailable. "
            "Install glslc or slangc, or pass --glslc"
        )

    def _compute_checksum(self, files: List[Path]) -> str:
        """Compute MD5 checksum of multiple files"""
        files_expanded = []
        for path in files:
            if '.' not in path:
                path = os.path.join(Path(__file__).parent / 'shaders', path)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        full_path = Path(root) / file
                        files_expanded.append(full_path)
            else:
                files_expanded.append(self.config.slang_src_path / path)

        hasher = hashlib.md5()
        for file_path in files_expanded:
            assert file_path.exists(), f"{file_path} does not exist"
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        with open(__file__, 'rb') as f:
            hasher.update(f.read())
        hasher.update(bytearray(self.config.slangc_compile_args, 'utf-8'))
        hasher.update(bytearray(self.config.glslc_compile_args, 'utf-8'))
        return hasher.hexdigest()

    def _should_compile(self, checksum_key: str, checksum: str) -> bool:
        """Check if compilation is needed based on checksum"""
        checksum_path = self.config.shader_dst_path / "checksum.json"
        try:
            with open(checksum_path, 'r') as fp:
                checksums = json.load(fp)
        except:
            checksums = {}

        if checksum_key in checksums and checksums[checksum_key] == checksum:
            return False
        
        checksums[checksum_key] = checksum
        with open(checksum_path, 'w') as fp:
            json.dump(checksums, fp, indent=4)
        return True
    
    def _compile_shader_slang(self, source_file: Path, job: ShaderJob,
                              target: Literal["glsl", "spirv", "host-callable"],
                              ext: Literal["comp", "spv"]) -> Tuple[bool, str]:
        source_file = str(self.config.slang_src_path / source_file)
        cmd = [str(self.slangc), source_file]
        job_name = job.name + '.' + ext
        output_file = self.config.generated_dst_path / job_name  # type: Path
        
        # Add defines
        for define, value in job.defines.items():
            cmd.extend([f"-D{define}={value}"])
        
        # Add target and other args
        cmd.extend(["-target", target])
        cmd.extend(self.config.slangc_compile_args.split())
        cmd.extend(["-o", str(output_file)])
        
        output = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output

    def _compile_shader_glsl(self, source_file: Path, job: ShaderJob, target: Literal["glsl", "spirv"], ext: Literal["comp", "spv"]) -> Tuple[bool, str]:
        source_path = Path(__file__).parent / 'shaders' / source_file / job.name
        job_name = Path(job.name).with_suffix('.'+ext)
        output_file = self.config.shader_dst_path / source_file / job_name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if self.glsl_via_slangc:
            cmd = [str(self.glslc), str(source_path)]
            for define, value in job.defines.items():
                cmd.extend([f"-D{define}={value}"])
            cmd.extend(["-target", "spirv", "-o", str(output_file)])
            return subprocess.run(cmd, capture_output=True, text=True, check=True)

        cmd = [str(self.glslc), str(source_path)]
        for define, value in job.defines.items():
            cmd.extend([f"-D{define}={value}"])
        if target is not None:
            cmd.extend(["-target", target])
        cmd.extend(self.config.glslc_compile_args.split())
        cmd.extend(["-o", str(output_file)])
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _compile_shader(self, source_file: Path, job: ShaderJob, target: Literal["glsl", "spirv"], ext: Literal["comp", "spv"]) -> Tuple[bool, str]:
        """Compile a single shader"""
        try:
            if '.' in source_file:
                job_name = job.name + '.' + ext
                output = self._compile_shader_slang(source_file, job, target, ext)
            else:
                job_name = os.path.join(source_file, job.name + '.' + ext)
                output = self._compile_shader_glsl(source_file, job, target, ext)
            if output.stdout != "" or output.stderr != "":
                # Slang/glslc diagnostics are warnings once the process exits 0.
                # Keep going so build_native is not blocked by non-fatal notes.
                return True, f"O Compiled {job_name} with warning: {output.stdout} {output.stderr}"
            return True, f"✓ Compiled {job_name}"
        except subprocess.CalledProcessError as e:
            return False, f"✗ Failed to compile {job_name}: {e.stderr}"
        except Exception as e:
            return False, f"✗ Error compiling {job_name}: {str(e)}"

    def _create_shader_jobs(self) -> List[Tuple[str, List[ShaderJob]]]:
        """Create all shader compilation jobs grouped by category"""
        jobs = []

        forward, backward = 0, 1  # export modes

        # Radix sort
        jobs.append(("radix_sort", [
            ShaderJob("upsweep.comp", {}),
            ShaderJob("spine.comp", {}),
            ShaderJob("downsweep.comp", {}),
        ], []))


        # FTGSPP 4D materialization, fused dynamic backward/Adam, and
        # fixed-capacity relocation.  Projection, tile sorting, and alpha
        # blending jobs below remain the original VkSplat shaders.
        jobs.append(("ftgspp_materialize.slang", [
            ShaderJob("ftgspp_materialize", {}),
        ], ["ftgspp_dynamic.slang", "config.slang", "utils.slang"]))
        jobs.append(("ftgspp_fused_backward_optimizer.slang", [
            ShaderJob("ftgspp_fused_backward_optimizer", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 3}),
            ShaderJob("ftgspp_fused_backward_optimizer_sh0", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 0}),
            ShaderJob("ftgspp_fused_backward_optimizer_sh1", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 1}),
            ShaderJob("ftgspp_fused_backward_optimizer_sh2", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 2}),
        ], ["ftgspp_dynamic.slang", "vertex_shader.slang", "strategy_utils.slang",
             "config.slang", "utils.slang", "spherical_harmonics.slang",
             "ftgspp_work_control.slang"]))
        jobs.append(("ftgspp_fused_dynamic_backward_adam.slang", [
            ShaderJob("ftgspp_fused_dynamic_backward_adam", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 3}),
            ShaderJob("ftgspp_fused_dynamic_backward_adam_sh0", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 0}),
            ShaderJob("ftgspp_fused_dynamic_backward_adam_sh1", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 1}),
            ShaderJob("ftgspp_fused_dynamic_backward_adam_sh2", {"EXPORT_MODE": backward, "ACTIVE_SH_DEGREE": 2}),
        ], ["ftgspp_dynamic.slang", "vertex_shader.slang", "strategy_utils.slang",
             "config.slang", "utils.slang", "spherical_harmonics.slang",
             "ftgspp_work_control.slang"]))
        jobs.append(("ftgspp_explicit_adam.slang", [
            ShaderJob("ftgspp_explicit_adam", {}),
        ], ["ftgspp_dynamic.slang", "strategy_utils.slang", "config.slang",
             "utils.slang", "spherical_harmonics.slang", "ftgspp_work_control.slang"]))
        jobs.append(("ftgspp_peer_mean.slang", [
            ShaderJob("ftgspp_peer_mean", {}),
        ], ["config.slang"]))
        jobs.append(("ftgspp_sparse_gradient.slang", [
            ShaderJob("ftgspp_sparse_flag", {"SPARSE_GRADIENT_PHASE": 0}),
            ShaderJob("ftgspp_sparse_pack", {"SPARSE_GRADIENT_PHASE": 1}),
            ShaderJob("ftgspp_sparse_scatter", {"SPARSE_GRADIENT_PHASE": 3}),
        ], ["config.slang"]))
        jobs.append(("ftgspp_regularizer_mean.slang", [
            ShaderJob("ftgspp_regularizer_mean", {}),
        ], ["ftgspp_dynamic.slang", "config.slang", "utils.slang"]))
        jobs.append(("ftgspp_relocation_prepare.slang", [
            ShaderJob("ftgspp_relocation_prepare", {}),
        ], ["ftgspp_relocation_binoms.slang", "strategy_utils.slang",
             "config.slang", "utils.slang"]))
        jobs.append(("ftgspp_relocation_copy.slang", [
            ShaderJob("ftgspp_relocation_copy", {}),
        ], ["strategy_utils.slang", "config.slang", "utils.slang"]))
        jobs.append(("ftgspp_relocation_plan.slang", [
            ShaderJob("ftgspp_relocation_score", {"RELOCATION_PHASE": 0}),
            ShaderJob("ftgspp_relocation_cdf", {"RELOCATION_PHASE": 1}),
            ShaderJob("ftgspp_relocation_sample", {"RELOCATION_PHASE": 2}),
            ShaderJob("ftgspp_relocation_cdf_normalize", {"RELOCATION_PHASE": 3}),
        ], ["config.slang", "utils.slang"]))
        jobs.append(("ftgspp_flow_features.slang", [
            ShaderJob("ftgspp_flow_features", {}),
        ], ["config.slang", "utils.slang"]))
        jobs.append(("ftgspp_hard_separation.slang", [
            ShaderJob("ftgspp_hard_separation", {}),
        ], ["config.slang"]))
        jobs.append(("ftgspp_loss_reduce.slang", [
            ShaderJob("ftgspp_loss_reduce", {}),
        ], ["config.slang", "ftgspp_work_control.slang", "ftgspp_image_objective.slang"]))
        jobs.append(("ftgspp_eval_metrics.slang", [
            ShaderJob("ftgspp_eval_metrics", {}),
        ], ["config.slang", "ftgspp_work_control.slang", "ftgspp_image_objective.slang"]))
        jobs.append(("ftgspp_color_correct_forward.slang", [
            ShaderJob("ftgspp_color_correct_forward", {}),
        ], ["config.slang", "utils.slang", "ftgspp_work_control.slang",
             "ftgspp_image_objective.slang"]))
        jobs.append(("ftgspp_color_correct_backward.slang", [
            ShaderJob("ftgspp_color_correct_backward", {}),
        ], ["config.slang", "utils.slang", "ftgspp_work_control.slang",
             "ftgspp_image_objective.slang"]))
        jobs.append(("ftgspp_color_adam.slang", [
            ShaderJob("ftgspp_color_adam", {}),
        ], ["config.slang", "ftgspp_work_control.slang"]))
        # SSIM
        jobs.append(("ssim.slang", [
            ShaderJob("ssim_forward", {"EXPORT_MODE": forward}),
            ShaderJob("ssim_backward", {"EXPORT_MODE": backward}),
        ], ["config.slang", "ftgspp_work_control.slang", "ftgspp_image_objective.slang"]))

        # Prefix Sum
        jobs.append(("cumsum.slang", [
            ShaderJob(
                f"cumsum_{prefix}{phase_name}",
                {
                    "CUMSUM_PHASE": phase_id,
                    **({"CUMSUM_VALUE_IS_FLOAT": 1} if prefix else {}),
                }
            )
            for prefix in ("", "f32_")
            for phase_name, phase_id in [
                ("block_scan", 1), ("scan_block_sums", 2),
                ("add_block_offsets", 3), ("single_pass", 0)
            ]
        ], []))

        # Alpha Blending
        tensor_bwd_configs = [(0, 8, 0), (0, 8, 8), (1, 16, 0)]
        jobs.append(("alphablend_shader.slang", [
            ShaderJob(f"rasterize_forward", {"EXPORT_MODE": forward}),
        ] + [
            ShaderJob(f"rasterize_backward_{i}", {
                "EXPORT_MODE": backward, "BACKWARD_MODE": min(i, 2),
                "USE_SUBGROUP_OPERATIONS": tensor_bwd_configs[max(i-2,0)][0],
                "SPLAT_BATCH_SIZE": tensor_bwd_configs[max(i-2,0)][1],
                "GROUP_REDUCE_BEFORE_ATOMIC": tensor_bwd_configs[max(i-2,0)][2],
            })
            for i in range(2+len(tensor_bwd_configs))
        ], ["config.slang", "utils.slang",
            "alphablend_shader_bwd_per_pixel.slang",
            "alphablend_shader_bwd_per_splat.slang",
            "alphablend_shader_bwd_tensor.slang",
            "ftgspp_work_control.slang"
        ]))

        # Tile Shaders
        jobs.append(("tile_shader.slang", [
            ShaderJob(f"generate_keys", {"EXPORT_MODE": forward, "ENTRY": 1}),
            ShaderJob("compute_tile_ranges", {"EXPORT_MODE": forward, "ENTRY": 2}),
        ], ["config.slang", "utils.slang", "ftgspp_work_control.slang"]))

        jobs.append(("ftgspp_tile_work_prepare.slang", [
            ShaderJob("ftgspp_tile_work_prepare", {}),
        ], ["config.slang", "ftgspp_work_control.slang"]))

        # Vertex Shader
        jobs.append(("vertex_shader.slang", [
            ShaderJob(
                "projection_forward",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 0,
                 "ACTIVE_SH_DEGREE": 3},
            ),
            ShaderJob(
                "projection_forward_sh0",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 0,
                 "ACTIVE_SH_DEGREE": 0},
            ),
            ShaderJob(
                "projection_forward_sh1",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 0,
                 "ACTIVE_SH_DEGREE": 1},
            ),
            ShaderJob(
                "projection_forward_sh2",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 0,
                 "ACTIVE_SH_DEGREE": 2},
            ),
            ShaderJob(
                "ftgspp_dynamic_projection",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 1,
                 "ACTIVE_SH_DEGREE": 3},
            ),
            ShaderJob(
                "ftgspp_dynamic_projection_sh0",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 1,
                 "ACTIVE_SH_DEGREE": 0},
            ),
            ShaderJob(
                "ftgspp_dynamic_projection_sh1",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 1,
                 "ACTIVE_SH_DEGREE": 1},
            ),
            ShaderJob(
                "ftgspp_dynamic_projection_sh2",
                {"EXPORT_MODE": forward, "FTGSPP_DYNAMIC_PROJECTION": 1,
                 "ACTIVE_SH_DEGREE": 2},
            ),
        ], ["config.slang", "utils.slang", "spherical_harmonics.slang",
             "ftgspp_dynamic.slang"]))

        return jobs

    def compile_all(self, force: bool = False) -> bool:
        """Compile all shaders"""
        job_groups = self._create_shader_jobs()
        all_successful = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:

            futures = {}

            for source_file, shader_jobs, deps in job_groups:
                if not shader_jobs:
                    continue

                checksum = self._compute_checksum([source_file] + deps)
                checksum_key = source_file#.rstrip('.slang')
                
                if not force and not self._should_compile(checksum_key, checksum):
                    continue
                
                print(f">>> Adding \"{source_file}\" to compilation queue")
                
                for job in shader_jobs:
                    if '.' in source_file:
                        futures[executor.submit(self._compile_shader, source_file, job, "spirv", "spv")] = None
                        # futures[executor.submit(self._compile_shader, source_file, job, "glsl", "comp")] = None
                    else:
                        futures[executor.submit(self._compile_shader, source_file, job, None, "spv")] = None

            for future in concurrent.futures.as_completed(futures):
                success, message = future.result()
                print(message)
                if not success:
                    all_successful = False

        if all_successful:
            manifest = [
                ("generated/" + job.name + ".spv") if "." in source
                else str(Path(source) / Path(job.name).with_suffix(".spv"))
                for source, jobs, _ in job_groups for job in jobs
            ]
            (self.config.shader_dst_path / "manifest.json").write_text(
                json.dumps({"schema": 1, "files": sorted(manifest)}, indent=2) + "\n",
                encoding="utf-8",
            )
        return all_successful


def main():
    parser = argparse.ArgumentParser(description="Parallel Slang shader compiler")
    parser.add_argument("--slangc", type=Path, help="Path to slangc executable")
    parser.add_argument("--glslc", type=Path, help="Path to glslc executable")
    parser.add_argument("--src", type=Path, default="./native/slang",
                       help="Source directory for Slang files")
    parser.add_argument("--dst", type=Path, default="./build/native/shaders",
                       help="Destination directory for compiled shaders")
    parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs")
    parser.add_argument("--force", action="store_true", help="Force recompilation of all shaders")
    
    args = parser.parse_args()
    
    config = CompileConfig(
        slang_src_path=args.src,
        shader_dst_path=args.dst,
        generated_dst_path=(args.dst / "generated"),
        slangc_path=args.slangc,
        glslc_path=args.glslc,
        max_workers=args.jobs
    )
    additional_args = ''.join(f" {key}={value}" for (key, value) in [
        # ('-DOPTIMIZATION_LEVEL', OPTIMIZATION_LEVEL)
    ] if value is not None)
    config.slangc_compile_args += additional_args
    config.glslc_compile_args += additional_args
    
    compiler = ShaderCompiler(config)
    success = compiler.compile_all(force=args.force)
    
    if success:
        print("✓ All shaders compiled successfully!")
        sys.exit(0)
    else:
        print("✗ Some shaders failed to compile!")
        sys.exit(1)


if __name__ == "__main__":
    main()
