# Omni Model Usage

This guide uses [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) as an example omni model with SGLang-Omni and the OpenAI-compatible API. Qwen3-Omni supports multi-modal input (text, image, audio) and can produce text-only or text + audio output depending on the mode.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

## Text-Only Mode

Text-only mode runs the thinker pipeline on a single GPU. It accepts multi-modal input (text, image, audio) and produces text output only.

### Launch the Server

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

For MMSU-style audio-input / text-output benchmarks with short requests, use
the fused text-path config so the full text path stays inside one worker
process:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_mmsu.yaml \
  --text-only \
  --port 8008
```

### Image and Text Input

Send an image with a text question to get a text response.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### Audio and Image Input

Send an audio file together with an image. The audio contains the spoken question ("How many cars are there in the picture?") and the model answers based on both inputs.

> **Note:** Set `"content": ""` (empty string) on the user message when all semantic content comes from audio, video, or images rather than text.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### Video and Audio Input

Send a video with a spoken audio question. The model watches the video, hears the question, and responds with text.

The Video-AMME CI benchmark uses this same modality combination: video input
plus a spoken question/options WAV, with only routing and answer-format
instructions in the text message.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## Speech Mode

Speech mode runs the full 9-stage pipeline across multiple GPUs. It produces both text (from the thinker) and audio (from the talker) output.

### Launch the Server

Speech mode can run as a colocated one-GPU worker using the colocated config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

Use `examples/configs/qwen3_omni_colocated_h200.yaml` on single-H200 workers.

For manual multi-GPU placement, use the example script:

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 1 \
  --port 8008
```

Or use the CLI without `--text-only` for the standard speech pipeline:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

By default, leave `mem_fraction_static` unset and let SGLang-Omni auto-size the
SGLang AR memory budget. If a specific machine needs manual tuning, you can pin
the value globally or per AR stage:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --mem-fraction-static 0.88
```

Use per-stage flags when the thinker and talker need different budgets:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --thinker-mem-fraction-static 0.88 \
  --talker-mem-fraction-static 0.88
```

The speech server launcher exposes the same per-stage controls:

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 1 \
  --port 8008 \
  --thinker-mem-fraction-static 0.88 \
  --talker-mem-fraction-static 0.88
```

`--mem-fraction-static` applies to both Qwen AR stages. Per-stage flags override
the global value for that stage. Values must be greater than `0` and less than
`1`.

### Rollout Stress Diagnostics

For RL-style workloads, first measure same-prompt rollout pressure before
changing memory fractions. The stress runner sends the same MMMU image prompt
with `N={1,2,4,8,16}` concurrent rollouts, keeps Thinker and Talker online by
default, and records request-level queue, CUDA memory, and KV-pool telemetry
through the event profiler.

```bash
python benchmarks/eval/qwen3_omni_rollout_stress.py \
  --base-url http://localhost:8008 \
  --rollout-counts 1,2,4,8,16 \
  --max-tokens 256 \
  --output-dir results/rollout_stress
```

The result summary is written to
`results/rollout_stress/rollout_stress_results.json`; profiler events are under
`results/rollout_stress/events` unless `--profile-event-dir` is set.

Set conservative overload guards when running near the memory limit:

```bash
SGLANG_OMNI_ADMISSION_MAX_WAITING=32 \
SGLANG_OMNI_ADMISSION_MIN_FREE_GPU_MEMORY_MB=4096 \
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

When the guard rejects a request, the client receives a controlled error
instead of a worker OOM or router 500.

## Single-GPU FP8 on H100/H20

SGLang-Omni can also serve native FP8 Qwen3-Omni checkpoints. Native FP8 uses
the checkpoint quantization config when loading the thinker and talker AR stages,
while keeping the same Qwen3-Omni request format shown below.

For one-GPU H100/H20 colocated launch, use the FP8 colocated config:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

The config file contains the FP8 checkpoint path:
`marksverdhei/Qwen3-Omni-30B-A3B-FP8`. You can still pass `--model-path` to
override the config value.

The FP8 path keeps dense FP8 GEMM on SGLang `auto` and defaults native FP8 MoE
to CUTLASS when supported. For Qwen3-Omni pipeline launches,
`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` is set as a default unless the operator has
already set that environment variable. This disables SGLang's all-M DeepGEMM
precompile session while keeping DeepGEMM available for dense FP8 GEMMs.

To opt back into SGLang's all-M DeepGEMM precompile behavior:

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

### Image and Text Input

Send an image with a text question to get both text and audio responses. Set `"modalities": ["text", "audio"]` to enable audio output.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### Audio and Image Input

Send an audio file with an image. The model hears the spoken question and sees the image, then responds with both text and audio.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### Video and Audio Input

Send a video with a spoken audio question. The model watches the video, hears the question, and responds with both text and audio.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## Request Parameters

The table below lists all parameters accepted by the `/v1/chat/completions` endpoint for Qwen3-Omni.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `null` | Model identifier |
| `messages` | list | (required) | List of chat messages, each with `role` and `content` |
| `modalities` | list | `["text"]` | Output modalities: `["text"]` for text only, `["text", "audio"]` for text and audio |
| `images` | list | `null` | List of image file paths (local paths or URLs) |
| `audios` | list | `null` | List of audio file paths (local paths or URLs) |
| `videos` | list | `null` | List of video file paths (local paths or URLs) |
| `max_tokens` | int | `null` | Maximum number of tokens to generate |
| `max_completion_tokens` | int | `null` | OpenAI-compatible alias for `max_tokens` |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Random seed for reproducibility |
| `stream` | bool | `false` | Enable streaming via SSE |
| `audio` | dict | `null` | Speech response format configuration, e.g. `{"format": "wav"}` |
| `stage_sampling` | dict | `null` | Per-stage sampling overrides, e.g. `{"thinker": {"temperature": 0.8}}` |

## Concurrency / prefill batching tuning (issue #760)

Under many concurrent streaming sessions the thinker can fragment prefill into
many tiny batches (size 1-2) instead of coalescing toward the running/token
budget the way vLLM continuous batching does. The knobs below (all **off by
default**, so existing behavior and CI thresholds are unchanged) help close the
gap. They are most relevant to a TP>1 thinker serving prefill-heavy streaming
workloads (e.g. live speech translation). See issue #760.

| Env var | Default | Effect |
|---|---|---|
| `SGLANG_OMNI_PREFILL_COALESCE_MS` | `0` (off) | On the idle->prefill transition, wait up to this many ms for more queued requests before scheduling the first prefill, so it is not size-1. Never stalls active decode (only engages when no decode batch / async step is in flight). |
| `SGLANG_OMNI_PREFILL_COALESCE_POLL_MS` | `1.0` | Poll granularity for the coalesce wait. The window runs a fixed `window/poll` iteration count, so all TP ranks stay in lockstep. |
| `SGLANG_OMNI_PREFILL_COALESCE_TARGET` | `max_running_requests` | Stop waiting early once this many requests are queued. |
| `SGLANG_OMNI_PREFILL_COALESCE_MIN` | `2` | Minimum queue depth before the window engages, so a lone request never pays the idle wait (avoids a low-concurrency latency tax). |
| `SGLANG_OMNI_PREFILL_COALESCE_MAX_TOKENS` | `chunked_prefill_size`, else `max_prefill_tokens` | Token budget: stop coalescing once queued prefill tokens reach this, so the coalesced batch never exceeds one (chunked) step. **Bounds the transient activation/broadcast memory — required to avoid a multimodal thinker-TP OOM at high concurrency (see Audio note below).** |
| `SGLANG_OMNI_CHUNKED_PREFILL_SIZE` | unset (chunking off) | Enable chunked prefill for the thinker so long audio prefills interleave with decode instead of blocking a whole step. The talker (projected-embeds) path is unaffected (chunking is force-disabled there). |
| `SGLANG_OMNI_TP_IDLE_BROADCAST_SKIP` | `false` | For TP>1, replace the per-loop `broadcast_pyobj` with a cheap int broadcast on idle loops (skips the pickle path when the inbox is empty). |
| `SGLANG_OMNI_LOG_PREFILL_STATS` | `false` | Periodically log average/max prefill batch size and queue depth, to quantify fragmentation before/after tuning. |
| `SGLANG_OMNI_PREFILL_STATS_INTERVAL_S` | `10` | Interval (seconds) for the prefill-stats log line. |

### Suggested starting point (2x A6000, thinker TP=2, ~32 streaming sessions)

```bash
export SGLANG_OMNI_PREFILL_COALESCE_MS=8
export SGLANG_OMNI_PREFILL_COALESCE_TARGET=32
export SGLANG_OMNI_CHUNKED_PREFILL_SIZE=4096
export SGLANG_OMNI_TP_IDLE_BROADCAST_SKIP=1
export SGLANG_OMNI_LOG_PREFILL_STATS=1
```

### Measuring before/after

```bash
# Concurrency sweep (records throughput_qps / latency / rtf)
python -m benchmarks.eval.qwen3_omni_rollout_stress \
  --base-url http://127.0.0.1:<port> \
  --rollout-counts 1,2,4,8,16,32 --max-tokens 256

# Streaming TTFT / time-to-first-chunk
python benchmarks/eval/benchmark_omni_streaming_ttft.py \
  --base-url http://127.0.0.1:<port> --label tp2 --repeats 5
```

Watch the `[prefill-stats]` log lines: `avg_batch` should rise toward
`max_running_requests` (and away from ~1) once coalescing is enabled.

### Audio / multimodal note (measured: 2x A6000, thinker TP=2, 30B base)

A **text** concurrency sweep shows coalescing lifts `avg_batch` from ~N/2 to ~N
with throughput/latency unchanged (text prefill is cheap / decode-bound). On a
**prefill-bound audio** sweep (~30s clips, ~400 prefill tokens/req) the trade-off
is different and worth knowing:

- coalescing also lifts `avg_batch` (~N/3 -> ~N), but throughput does **not**
  improve, because baseline continuous batching already saturates the thinker
  prefill (~12k prefill-tok/s on this hardware);
- with a count-only target and **no** token budget, coalescing accumulated 32
  audio requests (each carrying multimodal CUDA feature tensors that move through
  `broadcast_pyobj`) and **OOM-crashed `thinker_tp1`** at N=32;
- `SGLANG_OMNI_PREFILL_COALESCE_MAX_TOKENS` (default = `chunked_prefill_size`)
  fixes the crash (0 failures, throughput == baseline) and
  `SGLANG_OMNI_PREFILL_COALESCE_MIN` removes the low-concurrency latency tax.

Net: treat the coalesce window as **experimental**, most useful for bursty
arrival at or above the target concurrency. The always-safe wins are the thinker
TP `disable_custom_all_reduce` auto-fix, chunked prefill, the idle broadcast
skip, and the `[prefill-stats]` instrumentation (which is what surfaced the OOM
above).
