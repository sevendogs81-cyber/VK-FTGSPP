# Scene and foreground parameters: adaptive defaults

Start with the YAML from `ftgspp prepare dataset --pipeline scene|foreground`.
Choose `scene` for full-frame RGB reconstruction, or `foreground` for a masked
subject. Scene mode means full-scene reconstruction with the supported cameras;
it does not imply support for equirectangular/360-degree image projection.
Both modes accept `train.preset: preview|balanced|extended`, using different
reference recipes. Saved effective run plans retain their original settings.
See the [scene input routes](first-run.md#scene-mode-input-routes) or
[foreground input routes](first-run.md).

## Choose a training budget

Set one field, or pass the equivalent CLI override:

```yaml
train:
  preset: balanced
  iterations: null  # optional: clear a copied fixed count and use the preset
```

```bash
ftgspp train --config data/my_capture/ftgspp.yaml --set train.preset=preview
ftgspp train --config data/my_capture/ftgspp.yaml --set train.preset=extended
```

Foreground (`pipeline.mode: foreground`) main-phase budgets:

| `train.preset` | Intended use | 121 frames | 300 frames |
|---|---|---:|---:|
| `preview` | Check cameras, masks, geometry, motion and the training path | 35,000 | 87,000 |
| `balanced` (default) | First quality-oriented run | 70,000 | 174,000 |
| `extended` | Extra refinement when hold-out metrics still improve | 105,000 | 261,000 |

These are **training budgets**, not different reconstruction algorithms. The
foreground variants keep full input resolution, every-frame initialization and the same learning
rates. They do not increase Gaussian capacity. The optional
[preview example](../examples/foreground_preview.yaml) also sets `data.scale: 0.5`
for quicker image processing; selecting `train.preset: preview` alone does not.
Choose image scale before point preparation. Prepared manifests include the
image/camera geometry: changing scale, frame range, calibration or camera split
can invalidate them. Reprepare into a new directory with the matching config;
the half-resolution example uses a separate points path. Changing only the
training-budget preset keeps the same geometry inputs.

All three retain **2,000,000 total Gaussians for generated geometry**. For
`init.backend: points`, the default is the sum of the selected source PLY point
counts. Explicit numeric `init.num_gaussians` overrides either policy.

The reference horizon is informed by DNA v3 and 4DAnyone experiments on 121-frame
captures. The scaling rules and new presets are engineering heuristics, not a
cross-dataset quality benchmark. More training does not necessarily improve the
selected model: reserve a hold-out camera and keep `train.export_best: true`.

## Foreground adaptation

Let `F = data.frames.stop - data.frames.start`; `stop` is exclusive. For a preset
with reference budget `R` from the 121-frame column above:

```text
T = ceil_to_multiple_of_1000(R × max(0.5, F / 121))
```

This keeps approximately the same number of optimizer updates per timestamp.
The half-reference floor leaves short clips enough optimization time. There is
no hidden upper cap: inspect the plan for very long clips, then choose a smaller
budget or split the sequence if the cost is too high.

Default clocks are derived from the effective `T`, including an explicitly
overridden `train.iterations`:

| Field | Automatic rule | At `T = 70000` | At `T = 174000` |
|---|---|---:|---:|
| `train.relocation.start` | `min(T - 1, round(T / 35))` | 2,000 | 4,971 |
| `train.relocation.stop` | `ceil(T / 2)` | 35,000 | 87,000 |
| `train.relocation.every` | `max(1, round(T / 140))` | 500 | 1,243 |
| `train.checkpoint_interval` | `max(1, round(T / 35))` | 2,000 | 4,971 |
| `train.log_interval` | Same as the default checkpoint interval | 2,000 | 4,971 |

`round` uses Python's nearest-integer rule, with ties to even. Relocation stays
in the first half, with roughly the same number of events. Checkpoint selection
and optional optimizer snapshots run roughly 35 times, keeping validation and
storage costs bounded as the horizon grows. Explicit values for any of these
fields win independently; other clocks still derive from `T`. Changing only
`checkpoint_interval` does not change the independently resolved log interval.

The position and velocity final LR ratios remain `0.03`; their decay spans the
resolved optimizer horizon. Do not copy v8's very small terminal ratios from a
200k experiment into a shorter run.

Camera count does not multiply the default horizon. With the default flat view
sampler and a complete camera/time grid, each timestamp gets `1/F` of the view
exposures regardless of rig size; each **camera/time pair** gets fewer exposures
on larger rigs. Different visibility and angular complexity can still need more
training. Neither camera count nor image resolution alone predicts that cost.
Keyframe stride only changes initialization; training still sees all `F` frames,
so increasing stride does not shorten the automatic training horizon.

`train.batch_size: 4` is explicit opt-in. Presets count optimizer updates and
do not divide `T` by four: averaged gradients and optimizer trajectories are not
equivalent to four separate B1 updates. Use B1 for the reference recipe.

## Scene adaptation

Scene mode retains half-resolution training (`data.scale: 0.5`), mask-free
full-frame RGB, fixed black background, marginal gating, color correction (CC),
gate-included relocation scoring and hard separation (HS). It keeps its own
learning rates and `0.01` position/velocity final LR ratios. It does not inherit
the foreground gate/duration recipe. B1 remains the supported scene batch size.

The reference is 30k main steps on a 60-frame clip, as used in the existing
[SelfCap timing study](performance.md#selfcap-timing-comparison). This anchors
the schedule; the study used 500k Gaussians and does not validate the new 2M
starter, automatic FPS spacing, or quality on longer captures. For scene presets:

```text
T = ceil_to_multiple_of_1000(R × max(0.5, F / 60))
H = max(1, round(T / 10))  # separate hard-separation optimizer updates
```

| `train.preset` | Reference `R` / 60-frame main | 121-frame main + HS | 300-frame main + HS |
|---|---:|---:|---:|
| `preview` | 15,000 | 31,000 + 3,100 | 75,000 + 7,500 |
| `balanced` | 30,000 | 61,000 + 6,100 | 150,000 + 15,000 |
| `extended` | 45,000 | 91,000 + 9,100 | 225,000 + 22,500 |

At 300 frames the balanced default therefore performs **165,000 total updates**,
not 30,000. Use `--print-config` before a long run. `train.iterations` counts
only main-phase updates. Explicit main or HS budgets win independently; disabling
`train.hard_separation` skips the extra phase. Changing a preset alone keeps
the same image resolution, point sources and 2M generated-geometry budget.

| Scene field | Automatic rule | At `T = 30000` | At `T = 150000` |
|---|---|---:|---:|
| `train.relocation.start` | `min(T - 1, round(T / 60))` | 500 | 2,500 |
| `train.relocation.stop` | `ceil(5 × T / 6)` | 25,000 | 125,000 |
| `train.relocation.every` | `max(1, round(T / 300))` | 100 | 500 |
| `train.relocation.score_mode_start` | `min(T - 1, round(T / 10))` | 3,000 | 15,000 |
| `train.color_correction_start` | Same default warm-up fraction | 3,000 | 15,000 |
| `train.checkpoint_interval`, `.log_interval` | `max(1, round(T / 15))` | 2,000 | 10,000 |
| `train.hard_separation_steps` | `max(1, round(T / 10))` | 3,000 | 15,000 |
| `train.hard_relocation_steps` | `[floor(H / 2)]`, or `[]` when `H = 1` | `[1500]` | `[7500]` |

HS has its own local clock and starts from the selected main snapshot. Its
relocation is tied to `H`, including an explicitly supplied HS length. The
export is the **terminal HS model**; `export_best` selects the main snapshot
used for the handoff, not the best snapshot across HS. To compare quality, keep
`ckpt/main_selected.npz` as the soft-gate reference. More HS steps are not a
guarantee of better image quality. All explicit clock overrides and saved
resume clocks remain authoritative.

Scene keyframe spacing also adapts to FPS when `init.keyframe_stride` is omitted:

```text
stride = min(F, max(1, round(data.fps / 6)))
duration = stride / data.fps × init.duration_multiplier
```

This targets roughly one geometry keyframe every 1/6 second: stride 10 at
60 FPS, 5 at 30 FPS, 4 at 24 FPS. Short clips clamp to their frame count, and
the last frame is always included. It preserves temporal spacing across rates;
it does not detect motion. For fast motion use a smaller explicit stride (often
1), then regenerate the matching geometry. Do not grow stride merely to force
every long clip into the same keyframe count. Existing PLYs require their
**actual** stride, supplied with `prepare dataset --keyframe-stride N`.
The separate SelfCap adapter stages/link-selects a concrete PLY set using its
`--keyframe-stride` (default 10), so its generated YAML deliberately pins that
stride. Its training horizon still inherits the adaptive scene preset.

`init.da3.max_points: null` follows the actual keyframe count in both modes.
For 300 scene frames at 30 FPS, automatic stride 5 gives 61 keyframes, and the
2M budget yields a 120k dense DA3 cap per keyframe. The foreground 300-frame
example below instead uses 300 keyframes. Sequence length alone is insufficient
to interpret `max_points`.

Scene uses `init.backend: auto` by default: calibrated complete RGB PLYs are
reused, otherwise DA3 is needed (including static-rig camera estimation when
uncalibrated). This fallback keeps a 2M budget. For known user clouds, choose
`--points DIR` during dataset preparation or explicit `init.backend: points`;
then an omitted/null Gaussian budget counts selected PLYs. This also avoids an
incomplete user cloud silently falling back to a depth model. Supplied XYZ-only
clouds use `--method colorize` with image colors or gray, without foreground
masks. Do not apply silhouette carving/filtering to a mask-free full scene.

## `max_points`: per keyframe versus the total model

These controls have different units:

| Control | Scope | Default |
|---|---|---|
| `init.num_gaussians` | Total final initialization population across the clip | Generated: 2M; user `points`: selected PLY population |
| `init.da3.max_points` | Maximum fused **dense reference points per keyframe**, before Gaussian sampling | `null`: automatic |
| `prepare points --max-points N` | Explicit per-keyframe color-query/output cap; full geometry still tests occlusion | Omitted: budget-adaptive for generated geometry, uncapped for user clouds |
| `prepare points --quality` | Generated candidate headroom: `fast`, `balanced`, `high`, `full` | `balanced`: `4 × ceil(N / K)` |
| `prepare points --bounds-method` | Carving envelope: `auto` or `da3` | `auto`: per-frame silhouette frusta, with DA3 fallback |
| `prepare points --resolution N` | Carving grid cells along each of the three bound axes | `128`; does not specify output point count |

For the default `init.da3.quality: auto`, the existing DA3 rule is:

```text
K = ceil((F - 1) / init.keyframe_stride) + 1  # last frame is always included
B = max(1, floor(init.num_gaussians / K))
init.da3.max_points = min(8 × B, 120000)       # only when configured null
```

The cap bounds a dense geometry/nearest-neighbor reference, not the number of
Gaussians retained in that frame. Final Gaussian allocation also considers the
geometry's candidate population, spatial extent, shape and motion; it need not
be uniform across frames.

For a 2M total budget and stride 1:

| Clip | Average Gaussian budget per keyframe | Automatic DA3 dense cap per keyframe |
|---|---:|---:|
| 121 frames | About 16,529 | 120,000 |
| 300 frames | About 6,667 | 53,328 |

Explicitly setting `init.da3.max_points: 150000` keeps a 150k cap at both lengths.
It permits up to 18.15M versus 45M points across those dense caches; it does **not**
allocate 150k Gaussians to the whole sequence. Leave it `null` for adaptation.
Explicit `quality: fast|balanced|high` has different dense multipliers/caps;
retain `quality: auto` to use the rule above. A cap is a ceiling, not a promise
that reconstruction produces that many valid points.

Carving and generated point filtering now use a separate budget-adaptive color
plan. With `N = init.num_gaussians` and the actual selected `K` keyframes:

| Preparation quality | Maximum color queries per keyframe | 2M / 121 keyframes | 2M / 300 keyframes |
|---|---|---:|---:|
| `fast` | `2 × ceil(N / K)` | 33,058 | 13,334 |
| `balanced` (default) | `4 × ceil(N / K)` | 66,116 | 26,668 |
| `high` | `8 × ceil(N / K)` | 132,232 | 53,336 |
| `full` | All generated geometry | Uncapped | Uncapped |

These are preparation CLI options, independent of `train.preset` and
`init.da3.quality`. The default 4× headroom is a conservative engineering choice,
not a demonstrated universal reconstruction optimum. `--resolution` still controls
geometric detail; quality does not silently change that resolution or camera count.
All geometry supplies the z-buffer, while only spatially selected queries receive
multi-view colors. If too few queries have trusted colors, preparation retries
full coloring once and then caps trusted points. Untrusted generated points are
removed, so output population may be below the cap. L3 allocates the exact total
Gaussian budget afterward and uses the prepared clouds for velocity references.

`--max-points 150000` overrides this rule at every sequence length. Avoid copying
that literal when you want adaptation. `--quality full` disables the generated
cap for a density/quality comparison. User `--input-points` passed to
`point_filtering` or `colorize` has **no automatic cap**, regardless of quality;
only an explicit `--max-points` allows sampling. Filtering can still remove
inconsistent or uncolorable points. Carving with `--input-points` builds new
geometry and therefore uses the generated cap. `init.backend: points` counts
the selected output PLY population and freezes that number in the run plan.

Automatic carving bounds use each keyframe's calibrated silhouette boxes,
without assuming that a few sampled times cover the full motion. Complete valid
silhouettes are grouped according to the support threshold; their bounded-frustum
envelopes are combined with padding. Clipped/empty masks and views with validity
holes are excluded **from bounds only**, and remain in carving/coloring. If the
remaining rig cannot provide finite bounds, or the carved volume reaches the
automatic boundary, preparation falls back to full-sequence DA3 bounds. This is
intended for surrounding camera rigs; narrow-baseline/front-facing captures may
still need DA3. No camera calibration: estimate the static rig once at the first
keyframe, then try the same per-frame bounds. Keep the DA3 camera cache for reload.

Explicit `--bounds` or `--input-points` still bypass automatic bound estimation.
To reproduce the old dense preparation choices, use
`--bounds-method da3 --quality full` with the old `--resolution` and remaining
explicit geometry/color arguments. Existing prepared artifacts remain readable;
new generation writes v4 carving/filter manifests and v2 colorize manifests.
After preparation, the printed training command selects this output and preserves
the resolved generated Gaussian budget (or counts the resulting user cloud).
Each manifest records the resolved preparation plan, per-frame bounds, geometry
and query counts, fallback reasons, and stage timings. Read these before reducing
training iterations to address a preparation bottleneck.

**Equal total capacity cannot guarantee equal per-frame quality.** At a fixed 2M
total, a 300-frame clip has less initialization capacity per keyframe than a
121-frame clip. Adaptive training corrects underexposure, not this capacity loss.
For a similar capture, matching the average capacity of 2M/121 would require
about 4.96M Gaussians at 300 frames; that is a rough planning estimate, not an
automatic recommendation. If the memory cost is too high, train shorter clips
separately or index a temporally downsampled sequence with its correct FPS.
Increasing `keyframe_stride` is another tradeoff, but can hurt fast motion and
requires matching point preparation. More depth-cache points or optimizer
steps alone cannot replace missing model capacity.

## Key arguments and when to change them

Defaults in the following table describe foreground where the modes differ;
scene-specific defaults and phase semantics are in the section above.

| Argument | Units / default | Guidance |
|---|---|---|
| `data.frames.start`, `.stop` | Indexed frames; stop exclusive | Select the actual contiguous training interval. An incorrect span also gives an incorrect automatic budget. |
| `data.fps` | Frames/second; set to the actual indexed rate | Required for correct timestamps, motion and temporal support. It is not a quality knob. |
| `data.train_cameras`, `.eval_cameras` | Disjoint camera-ID lists | Reserve hold-out views before tuning. With no eval camera, the last model is exported. |
| `data.scale` | Image scale; foreground `1.0` | Try `0.5` for training image-memory pressure. It does not reduce model parameter memory. Compare metrics at a consistent evaluation resolution. |
| `init.backend` | Foreground `carving` | `carving`/`point_filtering` require matching prepared manifests; `points` explicitly selects user PLYs; `da3` invokes direct optional depth initialization. |
| `init.points_path` | Directory of `fXXXXXX.ply` | Must cover every selected keyframe including the last. Training consumes RGB; convert XYZ-only PLYs first. |
| `init.num_gaussians` | Total points / Gaussians | Leave inherited for generated geometry or `null` for user `points`. Increase only after checking capacity and memory; changing it changes Gaussian allocation and DA3 sizing. |
| `init.keyframe_stride` | Input frames; `1` | Keep `1` for fast motion. Larger values save preparation work but rely more on temporal interpolation. |
| `init.duration` | Seconds; `null` | Auto = `stride / fps × duration_multiplier`. Avoid copying a numeric duration between captures with different FPS/stride. |
| `init.duration_multiplier` | Dimensionless; `1.0` | Explicit temporal-support adjustment. Too large can blur motion; too small can leave temporal holes. |
| `init.num_velocity_nns` | Neighbor count; `3` | Local velocity initialization. Bad calibration/correspondences should be repaired before changing this. |
| `init.seed`, `train.seed` | Integers; `0`, `0` | Keep fixed while comparing one variable; repeat promising settings across seeds. |
| `init.da3.quality` | `auto` | Adapts depth resolution, fusion views and dense point cap. Requires the DA3 extra only when inference is actually needed. |
| `init.da3.process_res` | Image long-side target; `null` | Auto accounts for budget, native size and foreground coverage. It is separate from training `data.scale`. |
| `init.da3.max_views` | Views per depth inference chunk; `null` | Reduce for DA3 GPU-memory pressure. This is not the total fusion-camera count. |
| `init.da3.fusion_views` | Cameras fused per keyframe; `null` | Auto uses all cameras on small rigs and a spread subset on large rigs. Reducing it may weaken geometry. |
| `init.da3.voxel_size` | World-space length; `null` | Auto estimates neighbor spacing; avoid copying an absolute value across different scene scales. |
| `train.lrs.durations` | Temporal parameter LR; `0.001` | Conservative reference setting, not automatically calibrated to every FPS, motion or coordinate scale. Tune only with held-out temporal diagnostics. |
| `train.lrs.marginal_gates` | Gate LR; `0.0005` | `0` freezes gates; use only as a deliberate experiment. |
| `model.gate_init_logit` | Logit; `-1.0` | Keep for the common starter; v8's `-0.23` is an experimental alternative, not a dataset detector. |
| `train.lr_schedules.means`, `.velocity_model` | Final/initial LR ratios; `0.03` | Ratios over total optimizer steps, not absolute LRs. |
| `train.background.mode` | `random_solid` | Encourages foreground alpha separation across background colors. |
| `train.export_best` | `true` | Use held-out joint foreground PSNR+SSIM selection instead of assuming the final step is best. |
| `train.save_checkpoints` | `true` | Set false to save disk; exact optimizer resume then lacks periodic snapshots. Hold-out selection still runs. |
| `eval.frame_interval` | Frames; `1` | Dense temporal candidate pool. Raise only if validation cost is excessive. |
| `eval.candidate_max_views` | Candidate-scoring cap; `128` | Bounds periodic selection work. Full selected-model validation is separate. |

Spatial settings such as carving resolution, mask dilation in pixels, and depth
consistency tolerances should not be multiplied by sequence length. They depend
on geometry, calibration, image size and masks. Auto DA3 already considers
foreground coverage and native resolution; the training recipe does not infer
motion complexity, mask correctness or a safe GPU model size from frame count.

## Inspect and override safely

```bash
# Effective core YAML; no run directory, training, model inference or downloads.
ftgspp train --config data/my_capture/ftgspp.yaml --print-config

# Restore adaptive steps in a file that explicitly copied iterations: 70000.
ftgspp train --config data/my_capture/ftgspp.yaml --print-config \
  --set train.preset=balanced --set train.iterations=null

# Explicit controlled experiment. Unspecified clocks follow 90000 steps.
ftgspp train --config data/my_capture/ftgspp.yaml \
  --output runs/my_capture_90k --set train.iterations=90000

# Inspect the saved plan of an existing run; no new-run overrides allowed.
ftgspp train --resume runs/my_capture_90k --print-config
```

YAML paths resolve relative to the YAML file; paths passed via `--set` resolve
relative to the shell's working directory. Unknown keys fail instead of being
silently ignored. `train.iterations: null` is accepted for new runs in either
mode. To restore automatic relocation/eval clocks, remove their copied numeric
overrides from the YAML as well. Presets do not override explicit numbers.

`--print-config` resolves core settings and, for automatic user-cloud budgets,
reads the selected PLYs. It does not validate actual images, masks, calibration
or GPU fit. DA3 sizing fields can still be `null` here: scene-dependent values
are resolved during preparation/initialization and recorded in its cache
metadata. Training saves concrete core clocks and budgets in
`config.resolved.yaml`. Resume uses those saved values; changing current
defaults or a preset cannot extend or reinterpret an existing run.

## Workflow for a user or tuning agent

1. Inspect indexed frame count/FPS, camera splits and coordinate alignment;
   also inspect masks for foreground mode.
   Identify whether the source is raw RGB, XYZ clouds or complete RGB clouds.
   Use the matching [first-run route](first-run.md), with `point_filtering` for
   optional foreground filtering plus image coloring of supplied XYZ points.
   In scene mode use `colorize` instead; no foreground masks are required.
2. Print the effective plan. Record `F`, selected keyframe count, total Gaussian
   budget, steps, resolution, seeds, and the preparation manifest. Check the
   per-keyframe capacity before committing to a long sequence.
3. Run a preview to find failures. Do not interpret a half-resolution or
   smaller-budget run as a quality comparison against the full recipe.
4. Run balanced with a fixed hold-out. For scene mode use full-frame PSNR/SSIM
   and compare selected main versus terminal HS output. For foreground use `metrics.jsonl` and
   `full_validation_metrics.jsonl` plus rendered views to examine foreground
   PSNR/SSIM, silhouettes, foreground holes, background leakage and motion.
   Keep a locked final-test set out of parameter selection.
5. Change one cause at a time: geometry/masks first; capacity if detail is absent;
   horizon if the hold-out score is still improving; temporal support for
   motion holes/ghosting. A higher training loss score or sharper training-view
   render alone does not establish improvement on unseen views.
6. Keep the same split, metric resolution, seeds and initialization for controlled
   comparisons. Use a new output directory for every trial. Preserve each
   resolved config and selection record; report both selected-step and final-step
   behavior when investigating overfitting.

For memory failures, distinguish stages: lower training image scale for image
buffers, lower an explicitly chosen Gaussian budget for parameter/optimizer
memory, or lower DA3 `max_views`/`process_res` for depth inference. Preview's
shorter horizon alone does not lower peak model memory. For long clips at a
fixed capacity, compare a short-segment run before spending more steps on the
whole sequence.
