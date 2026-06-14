# Qwen3-Omni TP=2 concurrency: vLLM gap diagnosis (#760)

Measured diagnosis companion to issue
[#760](https://github.com/sgl-project/sglang-omni/issues/760). This directory is
self-contained evidence: the analysis, the raw A/B data, the SimulEval agent, and
the scripts to reproduce everything.

> **Headline:** for pure `Qwen3-Omni-30B-A3B-Instruct` (no-RAG, en→zh) on 2 GPUs
> (TP=2, 32 concurrent streaming sessions), vLLM does **12.97 seg/s** vs
> sglang-omni **8.71** at quality parity. The gap is **NOT** prefill
> fragmentation and **NOT** any GPU-side knob — it is **host-side per-turn
> latency** in the multi-process pipeline. Per-stage profiling: thinker stage
> 64%, encoder+aggregate queue 30%, **cross-process relay only ~2%**. The thinker
> is CPU-bound (100% CPU, GPU 60–75%) and the shared "pipeline" process is
> GIL-serialized. See [`FINDINGS.md`](./FINDINGS.md) and the full A/B log in
> [`COMPARISON.md`](./COMPARISON.md).

## Why this matters for #760

#760 hypothesized the gap was prefill-batch fragmentation. The prefill-coalesce
work does reduce fragmentation, but the eval below shows it only buys ~2% and
does not close the vLLM gap — because the bottleneck is off-GPU. Treat the
prefill PR as a small de-fragmentation improvement, not the parity fix.

## The SimulEval agent (`eval/remote_omni_agent.py`)

A SimulEval-native `SpeechToTextAgent` that delegates generation to a **remote
OpenAI-style engine**, so the exact same streaming policy A/Bs vLLM vs
sglang-omni with one agent:

- Every `--source-segment-size` ms of new audio → one `WriteAction`: the new
  audio increment (padded to Qwen's ~0.96 s minimum) is sent to the engine and
  its translation is emitted.
- Multi-turn chat carries prior **translations as text** (the audio increment is
  not resent), matching the demo's streaming `given_chunks` behavior.
- No-RAG: no `term_map` is attached (pure Qwen3-Omni).
- Engine wire formats: vLLM uses `input_audio` base64 content parts; sglang-omni
  uses top-level `audios` file paths. One flag (`--remote-engine`) switches.
- SimulEval records per-write source-time delays + computation-aware timing, so
  the `instances.log` is scored by FBK `stream_laal_term.py` for BLEU /
  StreamLAAL / StreamLAAL_CA.

An env-gated per-turn round-trip capture (`REMOTE_OMNI_LAT_DIR`) writes one
`.lat` file per worker for the latency analysis in `FINDINGS.md`.

## Reproduce

Paths in the scripts are environment-specific (Taurus); adjust `MODEL_PATH`,
`DATA_DIR`, `SPACY` (python with simuleval + soundfile + sacrebleu), and the FBK
`stream_laal_term.py` location.

```bash
# 0. Data prep: cut ACL6060 dev into per-segment wavs + SimulEval source/target
python eval/prepare_acl6060_segments.py   # -> $DATA_DIR/{seg}/segments.{source,target}

# 1. Launch ONE engine (TP=2, pure Qwen3-Omni-30B-A3B, no-RAG)
GPUS=2,3 PORT=8101 bash eval/servers/serve_sglang_qwen3omni.sh
#   or:  PORT=8200 bash eval/servers/serve_vllm_qwen3omni.sh

# 2. Concurrency sweep {1,8,16,32} + inline BLEU/StreamLAAL scoring
ENGINE=sglang BASE_URL=http://127.0.0.1:8101 OUT_ROOT=/path/runs/sglang_sweep \
  bash eval/run_sweep.sh
#   -> results.tsv (engine, N, seg/s, BLEU, StreamLAAL, StreamLAAL_CA)

# 3. Per-stage residency (sglang-omni built-in request-event profiler)
curl -X POST :8101/start_request_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"r","event_dir":"/path/events"}'
#   ... run an N=32 load via run_concurrency.py ...
curl -X POST :8101/stop_request_profile -H 'Content-Type: application/json' -d '{}'
python -m sglang_omni.profiler /path/events --format table   # stage + hop breakdown
```

## Files

| file | what |
|------|------|
| `FINDINGS.md` | curated diagnosis: gap, negative results, root cause |
| `COMPARISON.md` | full A/B log (every config, raw numbers) |
| `eval/remote_omni_agent.py` | the SimulEval remote agent (vLLM + sglang) |
| `eval/run_concurrency.py` | N parallel SimulEval workers + inline scoring |
| `eval/run_sweep.sh` | sweep a running engine over N and collate results |
| `eval/score_streamlaal.sh` | FBK `stream_laal_term.py` wrapper (BLEU/StreamLAAL) |
| `eval/prepare_acl6060_segments.py` | cut ACL6060 dev into SimulEval inputs |
| `eval/servers/serve_{sglang,vllm}_qwen3omni.sh` | TP=2 engine launchers |
| `scripts/sglang_omni_qwen3_text_tp_server.py` | repo-local launcher that builds the Qwen3-Omni **text** pipeline with thinker TP (the upstream text example does not expose thinker TP on the CLI) |

> Layout note: `eval/servers/serve_sglang_qwen3omni.sh` invokes
> `scripts/sglang_omni_qwen3_text_tp_server.py` relative to `REPO_ROOT`. To run
> from this package, set `REPO_ROOT` to this directory and `SGLANG_OMNI_SRC` to
> your sglang-omni checkout.

## The host-side lever we tested: per-stage processes (de-GIL) — REFUTED

The Qwen3-Omni **text** pipeline config (`sglang_omni/models/qwen3_omni/config.py`,
`_text_stages`) places **all** six stages in a single `process="pipeline"`; the TP
launcher only pulls `thinker` out into its own TP process group. So `preprocessing
+ image_encoder + audio_encoder + mm_aggregate + decode` share **one** OS process
— one GIL for all 32 concurrent streams. The **speech** pipeline already runs one
process per stage (`_SPEECH_DEFAULT_PROCESSES`), so the "de-GIL" fix is purely
topological: `scripts/sglang_omni_qwen3_text_tp_server.py --per-stage-processes`
(or `PER_STAGE_PROCESSES=1 bash eval/servers/serve_sglang_qwen3omni.sh`).

**Measured (within-node A/B on an idle node, N=32): it regresses, hard.**

| topology | seg/s | per-turn RTT (p50) | BLEU |
|----------|-------|--------------------|------|
| one `pipeline` process (stock) | **7.46** | **809 ms** | 32.0 |
| per-stage processes (de-GIL split) | 4.06 | 1702 ms | 32.4 |

Splitting **doubled per-turn RTT and cut throughput 46%**. The monolithic pipeline
process is already the better partition: separate processes convert cheap
in-process stage handoffs into cross-process relays that serialize large
multimodal payloads (audio features, encoder/merged embeddings) and add more
serial single-threaded stages — costing far more than the GIL they remove. So the
host-side cost is **not** relievable by re-partitioning; the gap is structural (see
`FINDINGS.md` §4). Full log in `COMPARISON.md`.
