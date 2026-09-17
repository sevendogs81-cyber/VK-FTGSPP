"""Stateless, checkpoint-reconstructible single-view sampling policies."""
from __future__ import annotations
from functools import lru_cache
from typing import Sequence
import numpy as np

MAX_SAMPLING_SEED = 0xFFFFFFFF
SAMPLING_PHASE_SEED_FORMULA = "phase_seed(main)=seed; phase_seed(hard_separation)=seed+train_iterations"


def sampling_phase_seed(
    *, seed: int, phase: str, train_iterations: int,
) -> int:
    """Return the stateless view-sampler seed for one optimizer phase."""

    seed = int(seed)
    train_iterations = int(train_iterations)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if seed > MAX_SAMPLING_SEED:
        raise ValueError("seed must fit uint32")
    if train_iterations <= 0:
        raise ValueError("train_iterations must be positive")
    if phase == "main":
        return seed
    if phase == "hard_separation":
        return (seed + train_iterations) & MAX_SAMPLING_SEED
    raise ValueError(f"unsupported sampling phase: {phase}")


@lru_cache(maxsize=8)
def _flat_epoch_permutation(
    seed: int, epoch: int, num_views: int,
) -> np.ndarray:
    """Return the immutable PCG64 permutation for one flat-view epoch.

    Keeping only the most recent epochs bounds host memory while avoiding a
    full ``num_views`` shuffle on every optimizer step.  The cache is an
    optimization only: the permutation is completely reconstructible from a
    domain-separated ``SeedSequence(seed, epoch, FEPM)`` and does not become
    checkpoint state. Keeping seed and epoch in separate entropy lanes avoids
    cross-run aliasing such as ``seed=7, epoch=1`` versus ``seed=8, epoch=0``.
    """

    entropy = [
        int(seed) & 0xFFFFFFFF,
        int(epoch) & 0xFFFFFFFF,
        0x4645504D,  # "FEPM": flat-epoch permutation domain.
    ]
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(entropy))
    )
    permutation = generator.permutation(int(num_views))
    permutation.flags.writeable = False
    return permutation


class _TorchMT19937:
    """Small CPU MT19937 engine matching PyTorch's RandomSampler randint path."""

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UMASK = 0x80000000
    _LMASK = 0x7FFFFFFF
    _MASK32 = 0xFFFFFFFF
    __slots__ = ("_state", "_left", "_next")

    def __init__(self, seed: int) -> None:
        state = [0] * self._N
        state[0] = int(seed) & self._MASK32
        for index in range(1, self._N):
            previous = state[index - 1]
            state[index] = (
                1812433253 * (previous ^ (previous >> 30)) + index
            ) & self._MASK32
        self._state = state
        self._left = 1
        self._next = 0

    @classmethod
    def _twist(cls, first: int, second: int) -> int:
        mixed = (first & cls._UMASK) | (second & cls._LMASK)
        value = mixed >> 1
        if second & 1:
            value ^= cls._MATRIX_A
        return value & cls._MASK32

    def _next_state(self) -> None:
        state = self._state
        self._left = self._N
        self._next = 0
        offset = 0
        for _ in range(self._N - self._M):
            state[offset] = (
                state[offset + self._M]
                ^ self._twist(state[offset], state[offset + 1])
            ) & self._MASK32
            offset += 1
        for _ in range(self._M - 1):
            state[offset] = (
                state[offset + self._M - self._N]
                ^ self._twist(state[offset], state[offset + 1])
            ) & self._MASK32
            offset += 1
        state[offset] = (
            state[offset + self._M - self._N]
            ^ self._twist(state[offset], state[0])
        ) & self._MASK32

    def random_uint32(self) -> int:
        self._left -= 1
        if self._left == 0:
            self._next_state()
        value = self._state[self._next]
        self._next += 1
        value ^= value >> 11
        value &= self._MASK32
        value ^= (value << 7) & 0x9D2C5680
        value &= self._MASK32
        value ^= (value << 15) & 0xEFC60000
        value &= self._MASK32
        value ^= value >> 18
        return value & self._MASK32

    def random_uint64(self) -> int:
        high = self.random_uint32()
        low = self.random_uint32()
        return ((high << 32) | low) & 0xFFFFFFFFFFFFFFFF


@lru_cache(maxsize=32)
def _torch_random_sampler_stream(
    seed: int, num_views: int, total_slots: int,
) -> np.ndarray:
    """Return PyTorch ``RandomSampler(replacement=True)`` flat indices.

    PyTorch 2.5 CPU ``RandomSampler`` yields chunks of
    ``torch.randint(high=n, size=(32,), dtype=int64, generator=...)``.  For
    ``n < 2**32`` that kernel consumes one MT19937 uint32 per sample and maps
    it with modulo.  We reproduce that tiny path directly so native training
    keeps its torch-free runtime dependency and pays only one cached startup
    cost per phase.
    """

    seed = int(seed)
    num_views = int(num_views)
    total_slots = int(total_slots)
    if seed < 0 or seed > MAX_SAMPLING_SEED:
        raise ValueError("torch_random_sampler seed must fit uint32")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if total_slots <= 0:
        raise ValueError("torch_random_sampler total_slots must be positive")
    generator = _TorchMT19937(seed)
    values = np.empty(total_slots, dtype=np.int64)
    if num_views >= 1 << 32:
        for index in range(total_slots):
            values[index] = generator.random_uint64() % num_views
    else:
        for index in range(total_slots):
            values[index] = generator.random_uint32() % num_views
    values.flags.writeable = False
    return values


def sample_rank_views(
    *, seed: int, step: int, world_size: int, num_views: int,
    sampling_policy: str = "flat_epoch",
    view_frames: Sequence[int] | None = None,
    view_cameras: Sequence[int] | None = None,
    sample_count: int | None = None,
) -> tuple[int, ...]:
    """Return deterministic, rank-addressable uniform view samples.

    Sampling is deterministic and rank-addressable rather than tied to mutable
    process RNG state, so scheduling order does not alter the selected views.
    One call returns one slot per rank in rank-major order and therefore cannot
    accidentally give every rank the rank-0 stream. ``flat_epoch`` lays
    optimizer-step/rank slots onto cached random permutations of the complete
    flat ``(frame, camera)`` view table. It therefore covers every view exactly
    once per epoch without forcing ranks onto the same synchronized frame.
    ``co_frame`` instead samples one frame, then assigns distinct cameras from
    that frame across ranks (with replacement only when fewer cameras exist
    than ranks). ``unique`` avoids duplicate views within a step when possible,
    while ``with_replacement`` matches Torch RandomSampler's
    uniform-with-replacement *distribution*, not its generator or bit-exact
    sample stream. ``torch_random_sampler`` reproduces FTGSPP's
    ``RandomSampler(replacement=True)`` index stream without importing torch;
    callers pass ``sample_count`` so the stream is cached once per phase.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    global_slots = int(world_size)
    if sampling_policy not in {
        "flat_epoch", "unique", "with_replacement", "co_frame",
        "torch_random_sampler",
    }:
        raise ValueError(
            "sampling_policy must be 'flat_epoch', 'unique', "
            "'with_replacement', 'co_frame', or 'torch_random_sampler'"
        )
    if sampling_policy == "torch_random_sampler":
        if step < 0:
            raise ValueError(
                "step must be non-negative for torch_random_sampler sampling"
            )
        if sample_count is None:
            raise ValueError(
                "torch_random_sampler requires sample_count so the legacy "
                "stream can be cached once per phase"
            )
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("torch_random_sampler sample_count must be positive")
        total_slots = sample_count * global_slots
        first_slot = int(step) * global_slots
        if first_slot + global_slots > total_slots:
            raise ValueError(
                "torch_random_sampler step is outside sample_count"
            )
        stream = _torch_random_sampler_stream(
            int(seed), int(num_views), int(total_slots)
        )
        return tuple(
            int(stream[first_slot + rank])
            for rank in range(int(world_size))
        )
    if sampling_policy == "flat_epoch":
        if step < 0:
            raise ValueError("step must be non-negative for flat_epoch sampling")
        values: list[int] = []
        for rank in range(int(world_size)):
            global_slot = int(step) * global_slots + rank
            epoch, offset = divmod(global_slot, int(num_views))
            permutation = _flat_epoch_permutation(
                int(seed), int(epoch), int(num_views)
            )
            values.append(int(permutation[offset]))
        return tuple(values)
    if sampling_policy == "co_frame":
        return _sample_co_frame_views(
            seed=int(seed),
            step=int(step),
            world_size=global_slots,
            num_views=int(num_views),
            view_frames=view_frames,
            view_cameras=view_cameras,
        )
    if world_size == 1:
        # Keep the historical keyed NumPy stream only for the two explicit
        # step-local legacy policies.  ``flat_epoch`` and ``co_frame`` must go
        # through their real implementations even for a single device.
        return (int(np.random.default_rng(int(seed) + int(step)).integers(num_views)),)
    # SeedSequence avoids collisions inherent in the old seed+step+rank form.
    entropy = [int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, 0x46544753]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    values = generator.choice(
        int(num_views),
        size=global_slots,
        replace=(
            sampling_policy == "with_replacement"
            or int(num_views) < global_slots
        ),
    )
    return tuple(int(value) for value in values)


def _sample_co_frame_views(
    *,
    seed: int,
    step: int,
    world_size: int,
    num_views: int,
    view_frames: Sequence[int] | None,
    view_cameras: Sequence[int] | None,
) -> tuple[int, ...]:
    """Bind ranks to one synchronized frame with distinct cameras when possible."""

    if view_frames is None or view_cameras is None:
        raise ValueError(
            "co_frame sampling requires view_frames and view_cameras metadata"
        )
    if len(view_frames) != num_views or len(view_cameras) != num_views:
        raise ValueError(
            "view_frames/view_cameras length must equal num_views for co_frame"
        )
    frames_by_id: dict[int, list[int]] = {}
    for index, frame in enumerate(view_frames):
        frames_by_id.setdefault(int(frame), []).append(int(index))
    frame_ids = sorted(frames_by_id)
    if not frame_ids:
        raise ValueError("co_frame sampling found no frames")
    entropy = [int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, 0x434F4652]  # COFR
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    frame = int(frame_ids[int(generator.integers(len(frame_ids)))])
    candidates = list(frames_by_id[frame])
    # Prefer diversity across cameras: shuffle then take unique cameras first.
    generator.shuffle(candidates)
    chosen: list[int] = []
    seen_cameras: set[int] = set()
    leftovers: list[int] = []
    for index in candidates:
        camera = int(view_cameras[index])
        if camera in seen_cameras:
            leftovers.append(index)
            continue
        seen_cameras.add(camera)
        chosen.append(index)
        if len(chosen) >= world_size:
            break
    while len(chosen) < world_size:
        pool = leftovers if leftovers else candidates
        chosen.append(int(pool[int(generator.integers(len(pool)))]))
    return tuple(int(value) for value in chosen[:world_size])
