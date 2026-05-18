---
name: sglang-omni-architecture
description: Architecture map and binding contracts for sglang-omni. Use BEFORE adding any new stage, scheduler, model, relay backend, or serve surface — to confirm where the change belongs and what invariants apply. Update this file when the architecture itself shifts (a new layer, a new scheduler shape, a renamed contract), not for incidental code changes.
---

# sglang-omni Architecture Skill

## Purpose

`sglang-omni` is a multi-stage pipeline runtime for omni-modal models (Qwen3-Omni, Fish S2-Pro, MiniMing-Omni, Voxtral-TTS). It composes upstream `sglang`'s AR primitives (TP layers, KV cache, prefill/decode) with project-owned pipeline orchestration (Stage / Coordinator / Relay) so that **encoders, AR LM, and vocoders can live on different processes / GPUs / parallelism configs**.

Before changing code, decide which architectural layer your change belongs to, and obey that layer's contract. If a change crosses layers, name the crossing explicitly in the PR.

## Top-level layout (verify with `ls sglang_omni/` before trusting)

```
sglang_omni/
├── cli/              # `sgl-omni` entry, argparse, config loading
├── client/           # async OpenAI-style Client + types (consumed by serve and tests)
├── config/           # PipelineConfig schema (pydantic), compiler, manager
│   ├── schema.py     # StageConfig, PipelineConfig, RelayConfig, EndpointsConfig
│   └── compiler.py   # _resolve_factory_args (BACKEND-RESOLUTION CONTRACT)
├── pipeline/         # Orchestration: Coordinator, Stage, mp_runner, relay_io
│   ├── coordinator.py     # Request lifecycle (PENDING→RUNNING→COMPLETED/FAILED)
│   ├── control_plane.py   # ZMQ control plane (small, latency-sensitive)
│   ├── mp_runner.py       # MultiProcessPipelineRunner, _build_tp_stage_specs
│   ├── relay_io.py        # extract_tensors / restore_tensors (data-plane boundary)
│   ├── stage/runtime.py   # Stage — IO shell that owns scheduler thread
│   ├── stage_process.py   # Child-process entry; CUDA env remap
│   ├── stage_group.py     # Process supervision; any_dead() probe
│   └── tp_control.py      # TPLeaderFanout + TPFollowerControlPlane (mp.Queue)
├── relay/            # Data-plane backends: shm / nccl / nixl / mooncake
├── scheduling/       # SCHEDULER LAYER — same inbox/outbox shape across variants
│   ├── messages.py            # IncomingMessage / OutgoingMessage (the contract)
│   ├── types.py               # Shared scheduling types
│   ├── omni_scheduler.py      # AR; composition-wraps upstream sglang.Scheduler
│   ├── simple_scheduler.py    # Non-AR single-fn (batched optional, max_concurrency)
│   ├── threaded_simple_scheduler.py  # Non-AR concurrent (CPU-bound / blocking)
│   ├── bootstrap.py           # create_sglang_infrastructure (AR worker bring-up)
│   ├── stage_cache.py         # Per-stage output cache
│   └── sglang_backend/        # Wrappers around upstream prefill/decode/KV cache
├── model_runner/     # ModelWorker, SGLangModelRunner, ThinkerModelRunner
├── models/           # Model-specific glue (per architecture)
│   ├── registry.py             # Auto-discovers per-model `config.py`
│   ├── qwen3_omni/             # Stages, payload_types, components/, ...
│   ├── fishaudio_s2_pro/       # Custom FishScheduler lives here, not in scheduling/
│   ├── ming_omni/
│   └── voxtral_tts/
├── preprocessing/    # Modality-specific preprocessing (text/image/audio/video)
├── proto/            # StagePayload + serialization helpers (cross-stage payload)
├── serve/            # FastAPI server: launcher, openai_api, protocol
├── profiler/         # Torch profiler control plane
└── utils/            # Misc (broadcast_pyobj re-export, etc.)
```

The `sglang_omni_v1/` directory **does not exist on main**: PR #435 retired V0 and renamed v1 from `sglang_omni_v1/` to `sglang_omni/`. Old branches and old docs still mention `sglang_omni_v1/`; ignore them and `ls` first.

## Two planes, kept separate (HARD INVARIANT)

| Plane | Carries | Backend | Owner |
|-------|---------|---------|-------|
| **Control plane** | `SubmitMessage`, `DataReadyMessage`, `AbortMessage`, `StreamMessage`, `CompleteMessage`, `ShutdownMessage`, `Profiler*` | ZMQ (`pipeline/control_plane.py`) | Stage's entry rank only |
| **Data plane** | Bulk tensor payloads | Relay (`relay/{shm,nccl,nixl,mooncake}.py`) | `pipeline/relay_io.py` |

**Never** put tensors through ZMQ. **Never** route control hops through the relay. `relay_io.extract_tensors` walks a `StagePayload` dict tree, pulls tensors into the relay, and replaces them with placeholders — the metadata pickle goes over ZMQ. `restore_tensors` reverses on the receive side.

## Request flow (end-to-end)

```
HTTP / OpenAI API (serve/openai_api.py)
  → Client.generate() (client/client.py)
    → Coordinator.submit() (pipeline/coordinator.py)
      → ZMQ control plane to entry Stage
        → Stage receives SubmitMessage / DataReadyMessage
          → input_handler (DirectInput | AggregatedInput) merges payloads
          → scheduler.inbox.put(IncomingMessage)
            scheduler.start() loop (dedicated thread):
              → build batch
              → forward (model_runner.execute / direct fn)
              → scheduler.outbox.put(OutgoingMessage)
          → Stage._drain_outbox routes results:
              terminal stage → CompleteMessage to coordinator
              else           → write to relay + DataReadyMessage to next stage
        → Coordinator collects CompleteMessages, resolves submit() future
```

## Stage / Scheduler / ModelRunner — three roles, three responsibilities

| Component | Role | Owns | Does NOT own |
|-----------|------|------|--------------|
| **Stage** (`pipeline/stage/runtime.py`, ~800 lines) | IO shell | control_plane, relay, input_handler, stream queue, scheduler thread | batching, forward, KV cache |
| **Scheduler** (`scheduling/*.py` and `models/*/...scheduler.py`) | Compute loop | inbox / outbox queues, batching policy, forward dispatch | sockets, relays, request lifecycle |
| **ModelRunner** (`model_runner/*.py`) | Stateless forward | `execute(batch)` — build ForwardBatch → prepare hooks → forward → post-hooks → sample → extract | scheduling, queueing |

**Rule:** A scheduler exposes exactly `inbox`, `outbox`, `start()`, `stop()`, `abort(request_id)`. Stage uses ONLY that surface. Don't add scheduler-type branching in Stage.

## Scheduler types (when to use which)

| Scheduler | File | Use for | Has KV cache? | TP-aware? |
|-----------|------|---------|---------------|-----------|
| `OmniScheduler` | `scheduling/omni_scheduler.py` | AR stages (thinker, talker, fish text2semantic) | Yes (via upstream sglang `Scheduler` composition + `__getattr__`) | Yes |
| `SimpleScheduler` | `scheduling/simple_scheduler.py` | Non-AR single-pass stages with optional batching (`max_batch_size`, `request_cost_fn`) and bounded `max_concurrency` | No | Single-rank only |
| `ThreadedSimpleScheduler` | `scheduling/threaded_simple_scheduler.py` | CPU-bound / blocking non-AR stages where `asyncio.to_thread` style concurrency makes sense | No | Single-rank only |
| `Code2WavScheduler` | `models/qwen3_omni/components/code2wav_scheduler.py` | Diffusion-style streaming code2wav | No | No |
| `FishScheduler` | `models/fishaudio_s2_pro/fish_scheduler.py` | Fish S2-Pro streaming acoustic + DAC vocoder | No | No |
| `QwenTalkerScheduler` | `models/qwen3_omni/talker_scheduler.py` | Subclass of `OmniScheduler` with talker-specific overrides | Yes | Yes |
| `StreamingDetokenizeScheduler` | `models/qwen3_omni/components/streaming_detokenizer.py` | Streaming detokenization stage | No | No |

**Rule:** Custom schedulers that are tied to a single model live in `sglang_omni/models/<model>/...`, not in `sglang_omni/scheduling/`. Only schedulers that are model-agnostic earn a slot in `scheduling/`.

## Stage roles: single / leader / follower

`Stage` (`pipeline/stage/runtime.py`) carries a `role: Literal["single", "leader", "follower"]` set by `_build_stage_groups`:

- `role="single"` — `tp_size == 1`. Stage owns ZMQ recv + relay reader + scheduler.
- `role="leader"` — TP rank 0 of a `tp_size > 1` stage. Same as single, plus a `TPLeaderFanout` that mirrors Shutdown / Profiler / Abort to followers via mp.Queue.
- `role="follower"` — TP rank > 0. **No ZMQ recv, no relay reader.** Control plane is `TPFollowerControlPlane` reading from a follower-side mp.Queue. Receives request data only via the scheduler's intra-rank broadcast.

Helper properties: `spec.owns_external_io`, `spec.is_leader`, `spec.is_follower`. Anywhere in `Stage` that touches external IO must guard on `self._owns_external_io`. `Stage._drain_outbox_follower` actively refuses to emit external traffic — this is load-bearing.

## Backend resolution contract (HARD INVARIANT)

Every launcher-side decision that branches on `backend` reads through exactly:

```python
_resolve_factory_args(stage_cfg, config).get("backend", "local")
```

`_resolve_factory_args` (`config/compiler.py`) merges:
1. `stage_cfg.factory_args` (PipelineConfig)
2. `config.runtime_overrides[stage_cfg.name]` (CLI / manual)

It deliberately does **NOT** consult the factory function's signature default. A future flip of a signature default from `"local"` to `"auto"` must not silently change launcher behavior; the launcher decides single-vs-multi-process pre-spawn, and signature defaults are not visible to it. To roll out a backend change for real, write `factory_args={"backend": "auto"}` into the config templates AND change the in-body `_resolve_backend("auto", ...)` helper — leave the signature default alone.

**Contract is NOT currently locked in by any regression test.** If you touch `_resolve_factory_args` or any launcher-side `backend` branch, add an explicit anchor test: construct a stage whose factory's signature default is `"auto"` but whose `factory_args` does not set `backend`, and assert the resolved value is what `factory_args + runtime_overrides` would yield — never `"auto"`.

## Per-stage TP launch (HARD INVARIANT — single-process-per-rank)

For `tp_size > 1` stages, `_build_tp_stage_specs` (`pipeline/mp_runner.py`):

1. Mints one `StageProcessSpec` per TP rank.
2. Allocates a unique NCCL port via `_NcclPortAllocator` (loopback TCP, base 29500).
3. Builds `follower_work_queues` / `follower_abort_queues` for leader→follower fan-out.
4. Tags rank 0 as `role="leader"`, others as `role="follower"`.

In the child process, **before importing torch**, `_prepare_cuda_environment` remaps `CUDA_VISIBLE_DEVICES` to the single physical GPU this rank owns and rewrites `factory_args["gpu_id"]=0`. Only one GPU is visible as `cuda:0`. For TP ranks with one visible device, also set `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true` so sglang's `GroupCoordinator` uses `device_id=0` instead of `device_id=local_rank`.

For local-host NCCL bring-up, set:
- `NCCL_SOCKET_IFNAME=lo`
- `NCCL_IB_DISABLE=1`
- `NCCL_P2P_DISABLE=1`

`MultiProcessPipelineRunner._monitor_children` polls `StageGroup.any_dead()` every 5s. On non-zero child exit, it calls `Coordinator.fail_all_pending(error)` so in-flight HTTP requests don't hang.

## SGLang upstream reuse boundaries

**Reuse from `sglang` main (don't reimplement):**
- TP-aware layers: `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `VisionAttention`, `MergedColumnParallelLinear`, `FusedMoE`.
- Distributed init: `init_distributed_environment`, `initialize_model_parallel`, `get_tp_group()` from `sglang.srt.distributed.parallel_state`.
- Broadcast helpers: `broadcast_pyobj` from `sglang.srt.utils`.
- AR runtime primitives: `Scheduler` (composition only, **never** inheritance — see `OmniScheduler`'s `__getattr__` strategy), `ScheduleBatch`, `ForwardBatch`, `ModelRunner`, `Req`, `SamplingParams`.
- Model loader: `DefaultModelLoader._get_all_weights` + `load_weights_and_postprocess`.
- Multimodal types: `MultimodalDataItem`, `Modality` from `sglang.srt.managers.schedule_batch`.
- `ServerArgs` — wrapped via `scheduling/sglang_backend/server_args_builder.py`.

**Project owns (don't push upstream yet):**
- Stage / Coordinator / mp_runner / Relay — multi-stage pipelines are project value-add, not an sglang concept.
- Scheduler shells — the common `inbox`/`outbox` shape.
- Per-stage TP launch — sglang itself only models a single TP process group.
- Custom non-AR schedulers (Code2Wav, Fish).

## Adding a new feature — workflow

Match your change to a layer, then follow that layer's rules.

### 1. New stage in an existing model

1. Add a `create_<your>_executor(...)` factory in `models/<model>/stages.py`.
2. Returns a callable (for `SimpleScheduler`-style stages) or a scheduler instance (for AR / custom).
3. Register the stage in the model's `config.py` `PipelineConfig` builder.
4. Wire `next` / `wait_for` / `merge_fn` in the PipelineConfig. Do NOT add launcher logic to "know" about it.
5. Provide payload shape via `payload_types.py` if it differs from existing types.
6. If TP > 1: confirm the factory accepts `tp_rank`, `tp_size`, `nccl_port` kwargs (TP preflight Layer 1 in `_build_stage_groups`).
7. If `backend` is a factory parameter: ensure default is `"local"` and any "auto" routing reads `factory_args["backend"]` rather than the signature default.

### 2. New model architecture

1. Create `sglang_omni/models/<your_model>/` with:
   - `config.py` — `make_pipeline_config(...)` builder + arch→config mapping for `registry.py`.
   - `stages.py` — factories that return callables or schedulers.
   - `payload_types.py` — `PipelineState` and per-stage payload dataclasses.
   - `components/` — per-stage compute units (preprocessor, encoders, talker, ...).
2. The model is auto-discovered by `models/registry.py:import_pipeline_configs` walking subpackages.
3. Tests live under `tests/test_model/<your_model>/` and `tests/unit_test/<your_model>/`.

### 3. New scheduler variant (model-agnostic)

Only justified if no existing scheduler fits. Must match the contract:

```python
class YourScheduler:
    inbox:  queue.Queue[IncomingMessage]
    outbox: queue.Queue[OutgoingMessage]
    def start(self) -> None: ...   # blocks calling thread; runs the busy loop
    def stop(self) -> None: ...    # signals start() to exit
    def abort(self, request_id: str) -> None: ...
```

Live in `sglang_omni/scheduling/<your_scheduler>.py`. If it's TP-aware, also document the two-channel broadcast pattern (metadata over CPU group via `broadcast_pyobj`, tensors over device group via `dist.broadcast`) and the three error domains (recoverable pre-forward / fatal forward / recoverable post-forward).

### 4. New relay backend

Live in `sglang_omni/relay/<your_backend>.py`. Match the `BaseRelay` interface in `relay/base.py`. Wire it into `pipeline/relay_io.py`'s backend dispatch. Add a `Literal` entry to `PipelineConfig.relay_backend`.

### 5. New serve surface (OpenAI-compatible route, etc.)

Add the route in `sglang_omni/serve/openai_api.py`. Use `Client.generate` / `Client.stream` to reach the coordinator. Do NOT touch the Coordinator or pipeline plumbing from a serve handler — only the client.

### 6. Touching the launcher / mp_runner

This is the most fragile layer. Required reading before changing:
- `pipeline/mp_runner.py:_build_tp_stage_specs`
- `pipeline/stage_process.py:_prepare_cuda_environment`
- `config/compiler.py:_resolve_factory_args` (the **Backend Resolution Contract** above)

Any change here must come with a regression test that locks the contract: assert that launcher-side `backend` resolution reads only `factory_args + runtime_overrides` and never falls back to the factory's signature default.

## DO / DON'T

**DO**
- `ls sglang_omni/` before writing code — memory may be stale; the filesystem wins.
- Match existing factory naming (`create_<stage>_executor`).
- Use `_resolve_factory_args` to read `backend` in launcher logic.
- Set `single_visible_device` on stages whose factory resolved to `"sglang"` or `"auto"` (encoder-TP migration).
- Use the relay for tensors, ZMQ for metadata. Always.
- Keep custom schedulers next to their model under `models/<m>/`.

**DON'T**
- Don't read factory signature defaults from launcher decisions — only `factory_args + runtime_overrides`.
- Don't inherit from upstream `sglang.Scheduler`. Compose + `__getattr__` instead (`OmniScheduler` pattern).
- Don't put tensors through ZMQ. Don't put control hops through relay.
- Don't branch Stage on scheduler type. The `inbox`/`outbox`/`start`/`stop`/`abort` shape is the contract.
- Don't emit external traffic from a follower stage. `_drain_outbox_follower` enforces this.
- Don't bypass TP preflight Layers 1 & 2 (`_build_stage_groups`). If the preflight rejects your stage, fix the factory; don't suppress the check.
- Don't put model-specific code in `scheduling/` or `pipeline/`. The model boundary is `models/<m>/`.

## In-flight workstreams (verify state before relying on these)

These were active as of 2026-05-17. Re-grep `git log origin/main` to confirm they've landed before assuming they're available.

- **Encoder TP partial-load (PR #375 / #423 family)** — adds `EncoderModuleSpec`, `EncoderModuleContainer`, `SGLangEncoderWorker`, a TP-aware encoder scheduler with two-channel broadcast + three error domains, and a `single_visible_device` flag on `StageProcessSpec`. Once landed, encoder stages will load only declared submodules (~5 GB vision encoder vs ~57 GB full thinker) and run with `tp_size > 1` via SGLang-native encoders. Phase 0 was opt-in via `factory_args["backend"]="sglang"`. Refactor commits renamed `Worker→Runner`, `Executor→Runner`, `leader/follower → entry/non-entry`. **Until merged, none of the above is on main.**
- **Real streaming (PR #406, merged)** — server-side streaming for text + audio in Qwen3-Omni V1; see `tests/unit_test/qwen3_omni/test_streaming.py`.
- **HF-parity patches (PR #434 / #436, merged)** — patches for the sglang Qwen3-VL vision encoder under `model_runner/_sglang_qwen3_vl_patches.py`.

When one of these merges, update the **Scheduler types** table and the **DO / DON'T** rules above; archive the corresponding bullet from this section.

## Updating this skill

This skill describes architecture, not features. Update it when:
- A layer is added, removed, or renamed (e.g., a new `relay/` backend type, the v1 rename of #435).
- A contract changes (e.g., the inbox/outbox shape, the backend-resolution rule).
- A new stage role appears (e.g., the encoder TP "entry / non-entry" rename when it lands).

Do NOT update it for:
- New stages within an existing model.
- New tests.
- Refactors that don't change layer boundaries.

A good rule of thumb: if your PR changes `pipeline/`, `scheduling/messages.py`, `config/compiler.py`, or `config/schema.py`, this skill probably needs a corresponding edit. Otherwise it doesn't.
