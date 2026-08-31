# Qwen3-TTS Silent-Bootstrap Research Log

## 2026-08-30 — T-PR16 capability check

### Hypothesis

Qwen3-TTS might emit one request-independent silent codec frame before speech. If the complete 16-codebook frame were invariant for a known checkpoint and sampling policy, serving could gate a bootstrap-frame optimization on that exact capability without using waveform RMS trimming.

### Experiment

- Source revision: `abc7640446629a837d043355ff7a756eb2a0190a`.
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- Hardware: one H200, BF16 weights, SDPA.
- Reference stack: `qwen-tts==0.1.1`, `transformers==4.57.3`, and `kernels==0.11.7` in an isolated persistent environment.
- Inputs: two x-vector-only requests and two ICL requests spanning short/long English text and short Chinese text, with three seeds per case.
- Sampling policies: shipped defaults, fully greedy, greedy semantic sampling only, and greedy subtalker sampling only.
- Probe: `benchmarks/eval/qwen3_tts_bootstrap_frame.py` with `benchmarks/eval/qwen3_tts_bootstrap_frame_base.json`.

The container image's project Transformers stack cannot load the upstream full reference model directly; SGLang uses its own Talker implementation there. The probe therefore isolates the upstream reference version instead of changing the serving environment.

### Result

| Sampling policy | Observations | Unique complete first frames | Single-frame RMS range |
| --- | ---: | ---: | ---: |
| Default | 12 | 11 | -89.57 to -37.67 dBFS |
| Fully greedy | 12 | 4 | -88.07 to -69.59 dBFS |
| Semantic greedy | 12 | 11 | -89.58 to -47.96 dBFS |
| Subtalker greedy | 12 | 7 | -88.07 to -33.06 dBFS |

Fully greedy generation was stable across seeds within each exact request, but the four request shapes produced four different complete frames. X-vector-only cases shared codebook-0 ID 1995 while their remaining 15 codebooks differed; ICL cases shared codebook-0 ID 109 while their remaining codebooks also differed. Under default sampling, a fixed seed still produced a different complete first frame for every tested request.

### Decision

Drop the frame-skipping implementation. The tested checkpoint has no request-independent complete bootstrap frame under either the shipped sampling policy or fully greedy generation, and several non-greedy first frames are not uniformly silent. A checkpoint-name or sampling-only gate would therefore be unsound. Do not add generic RMS trimming.

The raw observations are in `docs/benchmarks/qwen3_tts_bootstrap_frame_result.json` (SHA256 `1e590d38dc82fb895a3bfcae3ef307cc295054838dcbccdab652fcb690cddc9e`).
