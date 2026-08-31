# SPDX-License-Identifier: Apache-2.0
"""Probe whether Qwen3-TTS emits a request-independent first codec frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_model(
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str,
):
    from sglang_omni.models.qwen3_tts.compat import (
        apply_qwen_tts_transformers_compatibility_patches,
    )

    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
    )


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_custom_voice_case(wrapper: Any, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_ids": wrapper._tokenize_texts(
            [wrapper._build_assistant_text(case["text"])]
        ),
        "instruct_ids": [None],
        "languages": [case["language"]],
        "speakers": [case["speaker"]],
    }


def _custom_voice_codes(
    wrapper: Any,
    prepared: dict[str, Any],
    sampling: dict[str, Any],
) -> torch.Tensor:
    codes, _ = wrapper.model.generate(
        **prepared,
        non_streaming_mode=False,
        **sampling,
    )
    return codes[0].detach().cpu()


def _prepare_voice_clone_case(wrapper: Any, case: dict[str, Any]) -> dict[str, Any]:
    prompt = wrapper.create_voice_clone_prompt(
        ref_audio=case["ref_audio"],
        ref_text=case["ref_text"],
        x_vector_only_mode=bool(case.get("x_vector_only_mode", False)),
    )
    input_ids = wrapper._tokenize_texts(
        [wrapper._build_assistant_text(case["text"])]
    )
    ref_ids = [
        wrapper._tokenize_texts([wrapper._build_ref_text(case["ref_text"])])[0]
    ]
    prompt_dict = wrapper._prompt_items_to_voice_clone_prompt(prompt)
    return {
        "input_ids": input_ids,
        "ref_ids": ref_ids,
        "voice_clone_prompt": prompt_dict,
        "languages": [case["language"]],
    }


def _voice_clone_codes(
    wrapper: Any,
    prepared: dict[str, Any],
    sampling: dict[str, Any],
) -> torch.Tensor:
    codes, _ = wrapper.model.generate(
        **prepared,
        non_streaming_mode=False,
        **sampling,
    )
    return codes[0].detach().cpu()


def _decode_frame_metrics(wrapper: Any, frame: torch.Tensor) -> dict[str, Any]:
    frame = frame.to(device=wrapper.model.device, dtype=torch.long).unsqueeze(0)
    wavs, sample_rate = wrapper.model.speech_tokenizer.decode(
        [{"audio_codes": frame}]
    )
    wav = np.asarray(wavs[0], dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return {
        "samples": int(wav.size),
        "sample_rate": int(sample_rate),
        "rms": rms,
        "peak": peak,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    wrapper = _load_model(
        config["model_path"],
        config.get("device", "cuda:0"),
        config.get("dtype", "bfloat16"),
        config.get("attn_implementation", "sdpa"),
    )
    observations: list[dict[str, Any]] = []
    first_frames: dict[tuple[int, ...], torch.Tensor] = {}
    if config["mode"] == "custom_voice":
        prepared_cases = {
            case["name"]: _prepare_custom_voice_case(wrapper, case)
            for case in config["cases"]
        }
    elif config["mode"] == "voice_clone":
        prepared_cases = {
            case["name"]: _prepare_voice_clone_case(wrapper, case)
            for case in config["cases"]
        }
    else:
        raise ValueError(f"Unsupported mode: {config['mode']}")
    for sampling_name, sampling in config["sampling_configs"].items():
        for case in config["cases"]:
            for seed in config["seeds"]:
                _seed_all(int(seed))
                if config["mode"] == "custom_voice":
                    codes = _custom_voice_codes(
                        wrapper, prepared_cases[case["name"]], sampling
                    )
                else:
                    codes = _voice_clone_codes(
                        wrapper, prepared_cases[case["name"]], sampling
                    )
                first = codes[0].to(dtype=torch.long)
                frame_ids = tuple(int(value) for value in first.tolist())
                first_frames.setdefault(frame_ids, first)
                observations.append(
                    {
                        "sampling": sampling_name,
                        "case": case["name"],
                        "seed": int(seed),
                        "frame_count": int(codes.shape[0]),
                        "first_frame": list(frame_ids),
                    }
                )

    metrics = {
        ",".join(str(value) for value in frame_ids): _decode_frame_metrics(
            wrapper, frame
        )
        for frame_ids, frame in first_frames.items()
    }
    by_sampling: dict[str, Any] = {}
    for name in config["sampling_configs"]:
        rows = [row for row in observations if row["sampling"] == name]
        unique = {tuple(row["first_frame"]) for row in rows}
        by_sampling[name] = {
            "observations": len(rows),
            "unique_first_frames": len(unique),
            "constant": len(unique) == 1,
        }
    return {
        "model_path": config["model_path"],
        "mode": config["mode"],
        "by_sampling": by_sampling,
        "frame_metrics": metrics,
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["by_sampling"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
