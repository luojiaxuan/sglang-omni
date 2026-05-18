# SPDX-License-Identifier: Apache-2.0
"""Whitelist-based stage selection for pipeline configs.

Given a list of StageConfig and a whitelist of stage names, return a new list
containing only the whitelisted stages (plus any stage with required=True),
with all cross-stage edges (next, wait_for, stream_to, project_payload)
rewired to reference only stages still in the effective set.

A stage whose `next` would become empty after pruning falls back to its
declared `next_fallback` (with `project_payload_fallback` swapped in). If no
fallback is declared, a ValueError is raised naming the offending stage and
edge so the caller can fix either the whitelist or the model declaration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.config.schema import StageConfig


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _apply_stage_filter(
    stages: "list[StageConfig]", enabled_stages: list[str]
) -> "list[StageConfig]":
    """Return a new stage list filtered by the whitelist.

    Required stages (StageConfig.required=True) are auto-included regardless
    of whether the caller listed them. Edges are pruned and rewired according
    to per-stage fallback declarations. Raises ValueError with an actionable
    message if a stage's outbound edge collapses with no fallback available
    or if the caller named a stage that does not exist.
    """
    all_names = {s.name for s in stages}
    requested = set(enabled_stages)

    unknown = requested - all_names
    if unknown:
        raise ValueError(
            f"enabled_stages references unknown stages: {sorted(unknown)}. "
            f"Known stages: {sorted(all_names)}."
        )

    auto_included = {s.name for s in stages if s.required}
    effective = requested | auto_included

    result: list[StageConfig] = []
    for stage in stages:
        if stage.name not in effective:
            continue
        result.append(_rewire_stage(stage, effective))
    return result


def _rewire_stage(stage: "StageConfig", effective: set[str]) -> "StageConfig":
    updates: dict = {}

    # Prune wait_for
    if stage.wait_for is not None:
        kept_wait = [w for w in stage.wait_for if w in effective]
        if kept_wait != stage.wait_for:
            if kept_wait:
                updates["wait_for"] = kept_wait
            else:
                # All upstream removed; the fan-in collapses. Drop merge_fn too
                # since there is nothing to merge.
                updates["wait_for"] = None
                updates["merge_fn"] = None

    # Prune stream_to
    pruned_stream = [t for t in stage.stream_to if t in effective]
    if pruned_stream != stage.stream_to:
        updates["stream_to"] = pruned_stream

    # Prune next; apply fallback if it fully collapses
    if stage.next is not None:
        targets = _as_list(stage.next)
        kept_next = [t for t in targets if t in effective]

        if kept_next:
            new_next: str | list[str] = (
                kept_next[0]
                if isinstance(stage.next, str) and len(kept_next) == 1
                else kept_next
            )
            updates["next"] = new_next
            updates["project_payload"] = {
                k: v for k, v in stage.project_payload.items() if k in effective
            }
        else:
            # All `next` targets disabled — fall back if declared.
            if stage.next_fallback is None:
                raise ValueError(
                    f"Stage {stage.name!r}: all next targets {targets} are "
                    f"disabled by enabled_stages, and no next_fallback is "
                    f"declared on the stage. Either include one of "
                    f"{targets} in enabled_stages, or declare next_fallback "
                    f"(and project_payload_fallback if applicable) on the "
                    f"stage so the framework can rewire the DAG."
                )
            fallback_targets = _as_list(stage.next_fallback)
            missing_fb = [t for t in fallback_targets if t not in effective]
            if missing_fb:
                raise ValueError(
                    f"Stage {stage.name!r}: next_fallback {fallback_targets} "
                    f"references stages not in the effective set "
                    f"(missing: {sorted(missing_fb)}). Add the missing "
                    f"stages to enabled_stages, or revise the fallback "
                    f"declaration to use stages that will remain after "
                    f"filtering."
                )
            updates["next"] = (
                stage.next_fallback
                if isinstance(stage.next_fallback, str)
                else fallback_targets
            )
            # Replace project_payload entirely with the fallback projection.
            updates["project_payload"] = dict(stage.project_payload_fallback)

    if not updates:
        return stage
    return stage.model_copy(update=updates)
