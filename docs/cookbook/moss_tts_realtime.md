# MOSS-TTS-Realtime

[MOSS-TTS-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime)
is a 24 kHz streaming TTS model for low-latency voice agents. It uses a
Qwen3-1.7B global backbone, a four-layer frame-local Transformer, and 16 RVQ
codebooks from
[MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer).
The model supports 20 languages and accepts an optional reference clip for
zero-shot voice cloning.

| Component | SGLang-Omni implementation |
|---|---|
| Global backbone | SGLang Qwen3 paged attention, RadixAttention, and CUDA graph |
| Local Transformer | PyTorch SDPA with GQA, captured as one frame CUDA graph |
| Audio tokenizer | Causal MOSS-Audio-Tokenizer, 16 codebooks |
| Output | 24 kHz mono PCM/WAV |
| Concurrency | One active request |
| API | `/v1/audio/speech`, including streaming PCM |

## Launch

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Realtime \
  --config examples/configs/moss_tts_realtime.yaml \
  --port 8000
```

The model and codec share one GPU. On startup, SGLang-Omni captures batch-1
CUDA graphs for both the global backbone and the complete 16-codebook local
decode.

## Generate Speech

The reference clip may be a local path, HTTP URL, or base64 data URI. Local
paths require `--allowed-local-media-path` at server startup.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
    "voice": "default",
    "input": "Welcome to MOSS TTS Realtime.",
    "ref_audio": "/path/to/reference.wav",
    "response_format": "wav"
  }' \
  --output output.wav
```

Reference-free synthesis is also supported by omitting `ref_audio`.

## Streaming

Set `stream=true` and `response_format=pcm` to receive raw 24 kHz mono PCM:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Realtime",
    "voice": "default",
    "input": "This audio is returned as it is generated.",
    "ref_audio": "/path/to/reference.wav",
    "stream": true,
    "response_format": "pcm"
  }' \
  | ffmpeg -f s16le -ar 24000 -ac 1 -i pipe:0 output.wav
```

The default generation parameters match the upstream recommendation:
`temperature=0.8`, `top_p=0.6`, `top_k=30`,
`repetition_penalty=1.1`, and a 50-frame repetition window.

## Request-1 Performance

The reproducible driver is
`benchmarks/eval/benchmark_moss_tts_realtime_serving.py`. The comparison below
used one NVIDIA H200, the same model and codec revisions, the upstream
`prompt_audio1.mp3`, identical sampling settings, two warmups, and five
sequential measured requests. The native baseline was the upstream FastAPI
server with SDPA and `torch.compile(fullgraph=True)`.

| Backend | TTFA median | E2E mean | Audio mean | RTF mean |
|---|---:|---:|---:|---:|
| Upstream native | 299 ms | 3.722 s | 5.648 s | 0.660 |
| SGLang-Omni | **134 ms** | **1.490 s** | 5.632 s | **0.265** |

SGLang-Omni delivered 2.49x the real-time throughput and reduced median TTFA
by 55%. One native measured request delayed its first HTTP audio chunk until
request completion; it remains in the raw results. Full revisions, samples,
and commands are recorded in
`benchmarks/results/moss_tts_realtime_h200_request1.json`.

As a one-sample semantic smoke check, `openai/whisper-base.en` transcribed an
SGLang-Omni output exactly as the target sentence (WER 0.0). This is not a
model-quality benchmark.

```bash
python benchmarks/eval/benchmark_moss_tts_realtime_serving.py \
  --backend omni \
  --base-url http://127.0.0.1:8000 \
  --prompt-audio /path/to/MOSS-TTS/moss_tts_realtime/audio/prompt_audio1.mp3
```

## Current Limits

- Only one active request is supported, matching the upstream FastAPI server.
- The OpenAI-compatible endpoint is single-turn. Upstream multi-turn acoustic
  context sessions are not yet exposed.
- The repetition window is fixed at the upstream default of 50 frames.
