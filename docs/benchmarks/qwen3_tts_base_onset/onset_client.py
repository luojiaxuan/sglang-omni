"""Measure the leading silence of Qwen3-TTS Base voice-clone outputs.

Sends non-streaming /v1/audio/speech requests over a grid of
(mode, reference, prompt, seed), stores every WAV, and appends one JSON line
per request with the audible onset. Existing records are skipped, so a
relaunch continues where the previous run stopped.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

PROMPTS = [
    "Hello there, how are you doing today?",
    "Absolutely, let me check that for you right now.",
    "The quick brown fox jumps over the lazy dog.",
    "Seven.",
    "Thanks for calling, my name is Alex and I can help.",
    "Could you confirm your account number, please?",
    "It looks like your order shipped this morning.",
    "Let's schedule a follow up for next Tuesday.",
    "Could you help me?",
    "Can you repeat that?",
    "Would you help me?",
    "Please confirm your account number.",
]


def read_wav(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload)) as handle:
        assert handle.getsampwidth() == 2, handle.getsampwidth()
        sr = handle.getframerate()
        channels = handle.getnchannels()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    audio = pcm.reshape(-1, channels).mean(axis=1) / 32768.0
    return audio.astype(np.float32), sr


def peak_onset_ms(audio: np.ndarray, sr: int, threshold: float = 0.02, frame_ms: int = 5) -> float | None:
    # note (luojiaxuan): same detector as the original report, first 5 ms frame whose peak reaches 0.02 (-34 dBFS).
    frame = int(sr * frame_ms / 1000)
    n = len(audio) // frame
    peaks = np.abs(audio[: n * frame]).reshape(n, frame).max(axis=1)
    hits = np.flatnonzero(peaks >= threshold)
    return float(hits[0] * frame_ms) if hits.size else None


def rms_onset_ms(audio: np.ndarray, sr: int, dbfs: float = -40.0, window_ms: int = 10, hop_ms: int = 5) -> float | None:
    window = int(sr * window_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    limit = 10 ** (dbfs / 20)
    for start in range(0, max(len(audio) - window, 0) + 1, hop):
        if np.sqrt(np.mean(audio[start : start + window] ** 2)) >= limit:
            return float(start / sr * 1000)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18731")
    parser.add_argument("--model", required=True)
    parser.add_argument("--refs", required=True, type=Path, help="meta.lst of the reference clips")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--modes", default="xvec,icl")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    refs = {}
    for line in args.refs.read_text().splitlines():
        _, ref_text, wav, _ = line.split("|")
        refs[Path(wav).stem] = (str((args.refs.parent / wav).resolve()), ref_text)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wav").mkdir(exist_ok=True)
    records_path = args.out / "records.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            if record["status"] == 200:
                done.add(record["key"])

    jobs = []
    for mode in args.modes.split(","):
        for ref_id, (ref_path, ref_text) in sorted(refs.items()):
            for prompt_id, text in enumerate(PROMPTS):
                for seed in range(args.seeds):
                    key = f"{mode}|{ref_id}|{prompt_id}|{seed}"
                    if key not in done:
                        jobs.append((key, mode, ref_id, ref_path, ref_text, prompt_id, text, seed))
    print(f"{len(done)} done, {len(jobs)} to run", flush=True)

    lock = threading.Lock()
    session = requests.Session()

    def run(job: tuple) -> None:
        key, mode, ref_id, ref_path, ref_text, prompt_id, text, seed = job
        reference = {"audio_path": ref_path}
        body = {
            "model": args.model,
            "input": text,
            "seed": seed,
            "stream": False,
            "response_format": "wav",
            "x_vector_only_mode": mode == "xvec",
        }
        if mode == "icl":
            reference["text"] = ref_text
        body["references"] = [reference]
        start = time.perf_counter()
        response = session.post(f"{args.base_url}/v1/audio/speech", json=body, timeout=600)
        latency = time.perf_counter() - start
        record = {
            "key": key, "model": args.model, "mode": mode, "ref": ref_id,
            "prompt_id": prompt_id, "text": text, "seed": seed,
            "status": response.status_code, "latency_s": round(latency, 4),
        }
        if response.status_code == 200:
            audio, sr = read_wav(response.content)
            wav_path = args.out / "wav" / f"{mode}-{ref_id}-p{prompt_id:02d}-s{seed}.wav"
            wav_path.write_bytes(response.content)
            record.update(
                sr=sr,
                duration_s=round(len(audio) / sr, 4),
                onset_peak_ms=peak_onset_ms(audio, sr),
                onset_rms40_ms=rms_onset_ms(audio, sr),
                wav=wav_path.name,
            )
        else:
            record["error"] = response.text[:500]
        with lock:
            with records_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({k: record.get(k) for k in ("key", "status", "onset_peak_ms", "latency_s")}), flush=True)

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(run, jobs))


if __name__ == "__main__":
    main()
