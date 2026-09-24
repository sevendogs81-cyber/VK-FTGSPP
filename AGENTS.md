# Repository instructions

These instructions apply to the entire repository. A more deeply nested
`AGENTS.md`, if one is added later, may refine them for that subtree.

## Project contract

TurboGS is a Linux x86_64, Python 3.12, NVIDIA Vulkan implementation of
explicit-velocity 4D Gaussian training. The supported product surface is the
installed `ftgspp` command, documented YAML, one-device training, headless
rendering, and the optional local and remote viewers. Treat this repository as
an application and CLI rather than a general-purpose import library.

Preserve these boundaries:

- The first-release supported training path uses one Vulkan device and one view
  per optimizer update. Two-device synchronous code is retained for engineering
  work but is not a supported user workflow. Do not present it as production
  functionality.
- The ordinary CLI and training runtime are Torch-free. Torch, BiRefNet-HR,
  Depth Anything 3, LPIPS, CUDA preparation accelerators, TensorBoard, and the
  browser server are optional capabilities and must stay lazily imported.
- `pipeline.mode` selects the coordinated `scene` or `foreground` profile.
  Image objective and evaluation protocol are derived contracts, not separate
  public switches.
- A run resolves configuration once into an immutable effective plan. Resume
  reuses the saved plan and V6 checkpoint semantics; it must not silently
  reinterpret current defaults.
- Checkpoints, manifests, cache identities, camera paths, viewer sessions, and
  export-selection files are versioned protocols. Reject incompatible input
  explicitly. Change a schema or ABI identifier when semantics change, and add
  compatibility or rejection tests.
- Numerical equivalence, deterministic identities, exact clocks, sampler state,
  phase state, and overflow replay are correctness contracts. Do not trade them
  for apparent speed without evidence and an explicit format or algorithm
  revision where required.
- Generated data must be published atomically. Do not expose a partially written
  checkpoint, cache, model, manifest, screenshot, or video as complete.
- Preserve the research-only license and all third-party notices. Do not describe
  this project as OSI open source or commercially permissive.

## Canonical references

Read the smallest relevant set before changing behavior:

- `README.md`: supported product, quick start, system requirements, and public
  repository layout.
- `CONTRIBUTING.md`: maintainer constraints, generated-file policy, release
  boundary, and pinned-tool requirements.
- `docs/architecture.md`: runtime ownership, initialization, training phases,
  persistence, evaluation, and build boundaries.
- `docs/install.md`: dependency groups and supported installation modes.
- `docs/usage.md`: public CLI, data, training, foreground, evaluation, and
  optional-component behavior.
- `docs/viewer.md`: desktop, SSH browser, and headless viewer contracts.
- `docs/performance.md`: benchmark definitions; do not compare incompatible
  timing intervals.
- `docs/migration.md`: intentionally removed behavior and rejected legacy keys.
- `src/ftgspp/config/defaults.yaml`, `schema.py`, and `validate.py`: authoritative
  configuration defaults, types, derived fields, and hard validation.
- `pyproject.toml`, `uv.lock`, `native/CMakeLists.txt`, and
  `native/compile_shaders.py`: package, dependency, native build, and shader jobs.

Do not use local design notes as public specification. The ignored files
`docs/foreground_reconstruction_plan.md`,
`docs/initialization_pipeline_improvement_plan.md`, `docs/validation.md`,
`docs/viewer_plan.md`, `docs/viewer_performance.md`, `docs/viewer_rgba8.md`, and
`docs/viewer*_benchmark_*.json` are local research/QA material and are excluded
from distributions.

## Repository map

### Root and public entry points

- `pyproject.toml` defines the Python 3.12-only project, the `ftgspp` console
  script, optional extras, scikit-build-core configuration, and sdist/wheel
  contents.
- `uv.lock` is the reproducible dependency lock. Do not hand-edit it.
- `scripts/setup.sh` is the supported user installer. It pins uv 0.11.29 and
  Slang 2026.10 and supports the viewer, remote, initialization, and LPIPS
  options documented in `scripts/README.md`.
- `examples/` contains public scene and foreground YAML templates plus the
  locked final-test manifest example. Examples must continue to resolve in
  configuration tests.
- `LICENSE`, `LICENSES/`, and `THIRD_PARTY.md` define project and bundled-code
  licensing. Preserve them when changing dependencies or distribution contents.

### Python application: `src/ftgspp`

- `cli.py` is the installed dispatcher. Top-level commands are `doctor`,
  `train`, `eval`, `export`, `viewer`, and `render`; `prepare` lazily dispatches
  `dataset`, `masks`, `points`, `selfcap`, `final-test`, and `viewer`.
- `config/`
  - `defaults.yaml`: shipped configuration and comments.
  - `schema.py`: structured dataclass schema.
  - `compose.py`: resolvers, path resolution, profile defaults, immutable
    `ResolvedRunPlan`, and resolved-config persistence.
  - `validate.py`: static and resolved validation plus derived-field and
    world-size policy.
- `data/`
  - `multiview.py`: Torch-free camera/image manifest scanning, aliases, and
    synchronization.
  - `scene.py`: camera calibration, image/mask loading, hashing, undistortion,
    splits, lazy views, and final-test scene loading.
  - `prepare_dataset.py`: canonical `ftgspp-dataset-v1` manifests and starter
    YAML.
  - `prepare_selfcap.py`: video extraction and SelfCap layout adaptation.
  - `prepare_final_test.py`: independent locked final-test validation.
- `training/`
  - `runner.py`: fixed preflight → initialize/resume → main phase → optional
    hard-separation phase → selection/evaluation → export pipeline.
  - `backend.py`: thin NumPy wrapper around the Vulkan-owned training engine,
    RGB cache, and native camera/image boundary.
  - `initialize.py`, `init_plan.py`, and `point_sampling.py`: explicit Gaussian
    initialization, cache identities, DA3 scheduling inputs, fusion, sampling,
    velocity, and budget allocation.
  - `objective.py`, `foreground_metrics.py`, and `temporal_metrics.py`: reference
    objective and diagnostic math.
  - `background.py`: deterministic solid-background sampling.
  - `sampler.py`: checkpoint-reconstructible single-view sampling.
  - `distributed.py` and `distributed_contract.py`: in-development two-device
    orchestration and its production invariants.
  - `stage_timer.py`: explicit wall-clock stage accounting.
- `io/`
  - `model.py`: contiguous NumPy representation of native 4D Gaussians.
  - `checkpoint.py`: V6 explicit model/optimizer checkpoints, phase state,
    provenance, resume validation, and atomic latest pointer.
  - `export_selection.py`: selected model and matching color-correction state.
  - `ply.py`: point PLY input and compatibility 4DGS output. Viewer PLY is not a
    lossless native model because it omits dynamic gate state.
  - `init_cache.py`, `hashing.py`, and `provenance.py`: atomic cache publication,
    streaming hashes, and source identities.
  - `viewer_manifest.py`: portable rendering metadata beside the final model.
- `optional/`
  - `birefnet.py`: offline mask preparation; imports Torch only when invoked.
  - `da3.py`, `da3_schedule.py`, and `da3_evidence.py`: optional DA3 inference,
    camera conditioning, geometry checks, point fusion, schedules, and caches.
  - `visual_hull.py` and `point_color.py`: foreground carving/filtering and
    visibility-aware IBR coloring, with optional CUDA acceleration.
  - `lpips.py`: locked-final-test-only LPIPS extension.
  - `torch_compute.py`: lazy optional compute-device resolution.
- `viewer/`
  - `asset.py`: native NPZ and portable viewer-manifest validation.
  - `offline.py`: headless PNG sequence and optional FFmpeg MP4 export.
  - `path.py` and `session.py`: validated, deterministic, model-bound JSON.
  - `remote.py` and `remote.html`: single-client loopback HTTP/WebSocket viewer.
  - `app.py`: desktop/remote/render CLI glue and file I/O.

### Native runtime: `native`

- `CMakeLists.txt` builds the reusable `ftgspp_core`, the headless
  `_ftgspp_vk` Python extension, shader resources, native tests, and optionally
  `_ftgspp_viewer` when `FTGSPP_BUILD_VIEWER=ON`.
- `compile_shaders.py` owns the complete shader job matrix, defines, entry
  points, checksums, and `manifest.json`. Shader outputs belong in build or
  installed-resource directories.
- `src/device/`: Vulkan buffers, device/pipeline lifetime, submission, timing,
  telemetry, and pipeline-ring infrastructure.
- `src/render/`: projection/rasterization orchestration, adaptive scheduling,
  tile work control, reusable forward pass, and offscreen `ViewRenderer`.
- `src/training/`: Vulkan trainer state and training operation dispatch.
- `src/collective/`: session-scoped host mailbox and generation state for the
  in-development device pair.
- `src/bindings/`: pybind11 NumPy/Python contracts for training, offscreen
  rendering, and the optional desktop viewer.
- `src/viewer/`: GLFW/Dear ImGui presentation, direct F/T display, input,
  camera paths, screenshots, and overlays.
- `src/common/`: shared native configuration constants.
- `slang/`: all project shader source, including materialization, projection,
  rasterization, objectives, backward/Adam, relocation, color correction,
  evaluation, hard separation, viewer packing, and presentation.
- `shaders/radix_sort/`: bundled GLSL radix sort and its separate license.
- `tests/`: C++ pair-collective and GPU viewer packing/display executables.

Shader memory layouts are cross-language contracts. A buffer field, descriptor,
push constant, specialization define, work-control state, or shader entry-point
change must update Slang/GLSL, C++, bindings where exposed, compiler jobs, and
contract/numerical tests together.

### Tests and engineering tools: `dev`

- `tests/test_*.py`: the CPU/default contract suite. It covers configuration,
  public CLI, data manifests/caches, initialization, foreground behavior,
  reference gradients and metrics, native source contracts, checkpoints,
  distributed state, installation pins, and viewers.
- `tests/reference/`: independent numerical references used for comparison;
  never import these into installed runtime code.
- `tests/gpu/`: explicit GPU smoke, numerical, pair-equivalence, overflow replay,
  viewer, benchmark, and full-pipeline acceptance scripts. They require selected
  hardware and are not part of the default CPU gate.
- `tests/viewer_browser.test.cjs`: build-free browser behavior test for the
  embedded remote page.
- `tools/`: unsupported research and diagnostics. Official user evaluation is
  `ftgspp eval`; do not promote helpers to public behavior accidentally.

### Automation and documentation

- `.github/workflows/ci.yml`: locked CPU suite plus the native
  `pair_collective_test` build.
- `.github/workflows/artifacts.yml`: pinned clean sdist/wheel build and installed
  artifact/content audit; it does not publish.
- `.github/workflows/gpu.yml`: manually dispatched controlled NVIDIA acceptance,
  including the in-development pair path.
- `docs/README.md` indexes public documentation. Keep user-visible CLI/config
  changes synchronized with the relevant public page and README when the quick
  start or supported surface changes.

### Generated and local state

Do not edit or commit `.venv/`, `build/`, `dist/`, `runs/`, `reports/`, caches,
compiled `.so`/`.pyd`/`.spv` files, `src/ftgspp/shaders/`, or Python cache files.
Test and build commands may regenerate them locally. Do not delete unrelated
local outputs merely to obtain a clean tree.

## Implementation rules

### General workflow

1. Start with `git status --short`. Preserve user changes and distinguish them
   from edits made for the current task. Never reset, clean, checkout, or rewrite
   unrelated work.
2. Trace behavior from the public CLI/config through Python orchestration,
   persistence, bindings, C++, and shader code before editing a cross-layer
   feature.
3. Prefer the smallest coherent change. Remove obsolete paths only when the
   public contract explicitly removes them; compatibility behavior here is
   usually tested.
4. Add or update a meaningful regression test for bug fixes, numerical changes,
   persisted formats, security boundaries, and user-visible state transitions.
5. Update public docs for changes to commands, YAML, installation, outputs,
   viewer controls, supported hardware, or evaluation definitions.
6. Run the narrow relevant gate first. Stop once the appropriate broader gate
   passes unless a concrete remaining risk justifies more work.

Use existing style: direct Python with type hints where they clarify contracts,
NumPy arrays at native boundaries, RAII/lifetime-safe C++, and explicit schema
validation. Avoid new frameworks, dependency layers, configurable abstractions,
or duplicate execution paths for a single fixed pipeline.

### Configuration and CLI

- Keep defaults, dataclass schema, profile derivation, validation, examples,
  CLI overrides, resolved persistence, tests, and docs synchronized.
- Resolve filesystem paths with the same config-relative versus CLI-cwd rules
  already implemented in `compose.py` and `cli.py`.
- Reject unknown or retired settings. Do not restore migration-era switches
  listed in `docs/migration.md`.
- `doctor` must remain read-only and must not create a run directory.
- Resume accepts its saved effective plan. New-run overrides must not leak into
  resume semantics.

### Data, initialization, and optional models

- Preserve complete rectangular-grid and camera-split validation, strict
  synchronization, calibration geometry, pixel conventions, masks, alpha
  semantics, and content hashes.
- Cache reuse must be based on semantic inputs and completed markers. A source
  revision difference may be audit metadata when documented, but it must not
  hide an algorithm or data identity change.
- Never trigger model downloads, Torch import, CUDA initialization, or optional
  package import from `ftgspp --help`, ordinary scene training, viewer startup,
  or unrelated tests.
- Do not run real model downloads or large dataset preparation as routine
  verification. Use fixtures/mocks unless the user explicitly requests an
  acceptance run and supplies the required environment.

### Training, persistence, and evaluation

- Keep one forward loss consumable exactly once. Overflow retry must restore and
  replay the same generation/views rather than advance hidden state.
- Main and hard-separation phases have distinct local steps, seeds, moments, and
  checkpoint roles. Hard separation begins from the retained selected main
  snapshot, not the last main iteration.
- Model selection must use the documented hold-out joint PSNR+SSIM identity.
  A locked final-test split must never influence selection.
- LPIPS remains an explicit optional final-test metric and has no training YAML
  field.
- Preserve output identities such as `config.resolved.yaml`, `run.log`,
  `train_metrics.jsonl`, `metrics.jsonl`, `full_validation_metrics.jsonl`,
  `final_test_metrics.jsonl`, `ckpt/`, `gaussians.npz`, `export_selection.json`,
  `performance_summary.json`, the configured PLY export, and viewer metadata.
- Do not weaken hash, layout, sampler ABI, clock, optimizer, topology, or phase
  validation to make an incompatible checkpoint load.

### Native and shader work

- Python does orchestration and I/O; Vulkan owns training/model buffers and hot
  numerical work. Avoid unnecessary full model or image copies across the
  Python/native boundary.
- Maintain Vulkan synchronization, buffer generations, descriptor lifetimes,
  overflow latches, and bounded allocations. Do not infer GPU completion from
  host submission alone.
- Preserve FP32/FP64 choices used by reference contracts and document any
  precision change. Compare native results against independent references with
  explicit tolerances.
- Keep Slang 2026.10 compatibility. Do not commit compiled shaders.
- The default native build is headless. Do not introduce GLFW/X11/ImGui into
  `_ftgspp_vk`; desktop dependencies belong behind `FTGSPP_BUILD_VIEWER` in
  `_ftgspp_viewer`.

### Viewer work

- Camera poses use rigid OpenCV world-to-camera matrices: +X right, +Y down,
  +Z forward. Preserve intrinsics, field of view, original aspect ratio, and
  pixel-center conventions when resizing.
- Camera-path seconds and 4D scene seconds are independent. Preserve deterministic
  sampling, quaternion orientation interpolation, snapshots of export state, and
  fixed output FPS independent of interactive FPS.
- Faithful screenshots and exports rerender at the requested extent. Do not
  reuse reduced interactive JPEG frames. MP4 is explicitly lossy H.264/CRF18/
  yuv420p and opaque; PNG is the lossless/alpha-capable output.
- Remote mode remains loopback-only, token-protected, same-origin checked,
  single-client, display-free, and bounded in resolution/frame count. Do not add
  a frontend build system or server display requirement.
- State stored in camera `w2c` must flow consistently through session, bookmark,
  path, recording, screenshot, and video export code.

### Dependencies, packaging, and release files

- Python stays exactly 3.12.x. Base requirements must remain compatible with the
  Torch-free NumPy 2/OpenCV 4.12 runtime.
- Keep uv and Slang pins synchronized between `scripts/setup.sh`, workflows,
  tests, and documentation.
- Add a dependency only when existing standard-library/project facilities are
  insufficient. Put optional functionality in a matching extra and preserve
  lazy imports.
- If package contents change, update `pyproject.toml` include/exclude rules and
  artifact audits. Verify from an installed wheel outside the checkout when the
  import/resource boundary changes.
- Preserve third-party license files for pybind11, GLM, VkSplat, radix sort,
  GLFW, and ImGui as applicable to the build variant.

## Verification matrix

Commands assume the repository root. Use the existing environment when it is
already suitable. `uv sync` rewrites `.venv` and intentionally does not install
the project in the documented development mode, so do not run it casually in a
working environment.

### Fast and targeted checks

- Python test file:
  `.venv/bin/python -m pytest dev/tests/<test_file>.py -q`
- Specific test:
  `.venv/bin/python -m pytest dev/tests/<test_file>.py::<test_name> -q`
- Remote browser behavior:
  `node --test dev/tests/viewer_browser.test.cjs`
- Shell syntax after installer edits:
  `bash -n scripts/setup.sh`
- Whitespace/patch sanity:
  `git diff --check`

Choose tests by ownership:

- Config/CLI/examples: `test_config_validation.py`, `test_public_pipeline.py`,
  `test_cli_prepare.py`.
- Dataset/calibration/cache: `test_generic_dataset.py`,
  `test_dataset_manifest_cache.py`.
- Initialization/DA3/foreground: `test_da3_schedule.py`,
  `test_da3_initializer.py`, `test_initialization_regressions.py`,
  `test_initialize_sampling.py`, `test_foreground_reconstruction.py`.
- Training/reference math: `test_native_runtime.py`,
  `test_gpu_foreground_numeric.py`, `test_gpu_scene_cc_numeric.py`,
  `test_exact_tile_culling.py`.
- Distributed/pair: `test_distributed_contract.py`,
  `test_distributed_runtime.py`, `test_pair_collective_reference.py`.
- Checkpoints/exports: `test_checkpoint_v6.py`, `test_export_selection.py`.
- Viewer: `test_viewer.py`, `test_remote_viewer.py`, and
  `viewer_browser.test.cjs`.
- Native ABI/source layout: `test_native_contracts.py` plus the closest numerical
  test.
- Installer/pins: `test_install_pins.py` and `bash -n scripts/setup.sh`.

### Broader CPU gate

The CI-equivalent Python command is:

```bash
uv run --no-sync pytest -q
```

In an already prepared local environment, this equivalent avoids environment
changes:

```bash
.venv/bin/python -m pytest -q
```

The suite currently collects hundreds of tests; use it for cross-cutting
Python/config/persistence changes or before a release candidate, not for every
small documentation edit.

### Native build and C++ contracts

For a fresh headless contract build, follow CI:

```bash
.venv/bin/cmake -S native -B build/contracts -G Ninja \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$PWD/.venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
.venv/bin/cmake --build build/contracts --target pair_collective_test
.venv/bin/ctest --test-dir build/contracts --output-on-failure
```

For C++ or shader changes, rebuild the affected native target in an existing
configured build directory or run `uv build --python 3.12`. Desktop changes
also require a build configured with `FTGSPP_BUILD_VIEWER=ON`. A successful
compile is required even when Python source-contract tests pass.

### GPU acceptance

GPU scripts are explicit acceptance checks. Run only the smallest relevant
script with Vulkan device IDs from `ftgspp doctor`; CUDA ordinals may differ.
Examples include:

- `dev/tests/gpu/smoke_native.py`
- `dev/tests/gpu/check_viewer.py`
- `dev/tests/gpu/check_viewer_rgba8.py`
- `dev/tests/gpu/check_remote_viewer.py`
- `dev/tests/gpu/check_pair_equivalence.py`
- `dev/tests/gpu/check_composite_overflow_replay.py`
- `dev/tests/gpu/smoke_sync_multi_gpu.py`
- `dev/tests/gpu/run_pipeline_acceptance.py`

Do not run multi-device, long training, benchmark, model-download, or dataset
acceptance jobs merely because they exist. They require suitable controlled
hardware, data, time, and an explicit reason. Report the exact checks run and
state clearly when GPU or quality acceptance was not performed.

## Completion standard

Before declaring work complete:

- Review the final diff and status; include only task-related source changes.
- Confirm public docs and examples match final behavior.
- Confirm persisted-format and optional-dependency boundaries remain explicit.
- Run the relevant checks above and report actual results, including skips or
  environmental limitations.
- Do not claim reconstruction quality, numerical parity, performance gains, or
  GPU compatibility from CPU tests or successful compilation alone.
