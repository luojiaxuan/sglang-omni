# Qwen3-Omni talker partial-start — opt-in benchmark

Issue #473 (AC-10) requires a manual benchmark on ion8-h200-omni and
ion9-h200-omni to demonstrate that, with the partial-start knob enabled, the
talker stage begins prefill before the thinker stream completes, and that
audio TTFT improves while WER stays within tolerance.

This harness is **opt-in** and is **not** collected by pytest in CI.

## Container provisioning

Per the existing H200 conventions:

```bash
HOST=ion8-h200-omni      # repeat for ion9-h200-omni

ssh "$HOST" <<'BASH'
set -euo pipefail
NAME=sglang-omni-hayden
if docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}' | grep -qx "$NAME"; then
    NAME=sglang-omni-hayden-dev
fi
docker run -d --gpus all --name "$NAME" \
    --shm-size=64g --ipc=host --ulimit memlock=-1 \
    -v "$HOME":/sgl-workspace \
    -p 8000:8000 \
    frankleeeee/sglang-omni:dev sleep infinity
docker exec "$NAME" bash -lc \
    'cd /sgl-workspace && git clone https://github.com/sgl-project/sglang-omni.git || true'
echo "container ready: $NAME"
BASH
```

Inside the container, install this branch's working tree and any benchmark
requirements (`httpx` is the only extra dependency this harness needs).

## Server bring-up — two configurations

The plan compares two configurations per host:

1. **disabled** (baseline):

    ```bash
    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000
    ```

2. **enabled** (partial-start, decode-ready operating point):

    ```bash
    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000 \
        --runtime-override talker_ar:partial_start_min_chunks=5
    ```

   (Or edit the model's `_talker_stage` factory args in `config.py` for a
   sticky override.)

Use the detached-pytest pattern (redirect stdout/stderr to a log file) when
launching the server to avoid the documented pytest+NCCL+pipe deadlock.

## Measure TTFT

For each configuration, run:

```bash
mkdir -p results
python -m tests.perf_smoke.qwen3_omni_partial_start.measure_ttft \
    --base-url http://127.0.0.1:8000 \
    --label disabled \
    --output results/${HOSTNAME}-disabled.json \
    --repeats 3

# Restart server with the enabled config, then:
python -m tests.perf_smoke.qwen3_omni_partial_start.measure_ttft \
    --base-url http://127.0.0.1:8000 \
    --label enabled-min5 \
    --output results/${HOSTNAME}-enabled-min5.json \
    --repeats 3
```

Each invocation runs the **short** and **medium** synthetic prompts three
times and writes per-run TTFT, total stream time, and body-byte counts.

## Acceptance evidence (AC-10)

For each host, collect:

1. The two JSON result files from above.
2. The server stderr/stdout log containing the per-build observability lines
   emitted by `request_builders.py` (look for
   `talker_request_build request_id=... thinker_chunks=...`). The enabled run
   should show these emitted while the thinker stream is still in flight.
3. The corresponding SeedTTS WER run from
   `benchmarks/eval/benchmark_omni_seedtts.py` on a small subset
   (`--max-samples 50` is enough), once disabled and once enabled.

Document the comparison in `results/<host>/summary.md`:

- Enabled-vs-disabled TTFT delta per prompt (with stdev).
- Whether at least one prompt shows talker `talker_request_build` before
  thinker `stream_done` in the server log.
- WER delta vs the default-disabled bar (target: ≤ 1.0 percentage point per
  DEC-2 default).

## Non-CI status

The script does not import any test framework, and the package
`tests/perf_smoke/qwen3_omni_partial_start/` is excluded from pytest
collection by virtue of its location outside `tests/unit_test/` and
`tests/test_model/`. If a future change adds pytest collection to
`tests/perf_smoke/`, this harness must remain env-gated so CI does not depend
on H200 hardware.
