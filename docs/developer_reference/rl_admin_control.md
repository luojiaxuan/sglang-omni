# RL Admin Control

SGLang-Omni exposes a small administrative API for inference-side RL workflows.
The contract follows the SGLang and Miles control surface while preserving the
Omni pipeline boundary:

```text
HTTP / router -> Client -> Coordinator -> Stage -> Scheduler -> ModelWorker
```

The control plane carries only metadata and small result summaries. Tensor
payloads and bulk checkpoint data must be moved through disk, a distributed
group, or another data plane.

## Worker Endpoints

The worker server supports:

- `GET|POST /model_info`
- `POST /pause_generation`
- `POST /continue_generation`
- `POST /update_weights_from_disk`
- `POST /update_weights_from_tensor`
- `POST /update_weights_from_distributed`
- `GET|POST /weights_checker`

`/update_weights_from_disk` is the primary implemented update path. It pauses
the target scheduler, optionally aborts active requests, calls the underlying
SGLang model runner update method, optionally flushes cache, and resumes unless
`keep_pause=true`.

`/update_weights_from_tensor` and `/update_weights_from_distributed` are wired
as hooks. Tensor updates reject serialized tensors in the admin payload because
the admin control plane is not a tensor data plane.

## Stage and TP Behavior

The Coordinator sends one admin operation to each target stage and waits for
stage results. For TP stages, rank 0 fans the operation out to follower ranks,
collects one result per rank, and returns a stage-level aggregate result with
`rank_results`.

Stages without an admin-capable scheduler return a successful skipped result so
mixed pipelines can broadcast model info or pause commands without failing on
pre/post-processing stages.

## Router Behavior

The external router broadcasts admin requests to every non-dead worker. Update
and pause routes temporarily disable target workers from normal request routing
while the broadcast is in flight, then restore each worker's previous disabled
state.

## Weight Checker

`/weights_checker` supports `snapshot`, `reset_tensors`, `compare`, and
`checksum`. The Omni checker computes strict SHA256 digests from each tensor's
name, dtype, shape, and raw bytes, then derives a per-rank checksum from the
sorted tensor digests.
