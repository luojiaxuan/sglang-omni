# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import (
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    parse_message,
)
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingCoordinatorControlPlane,
    RecordingStageControlPlane,
)


class AdminScheduler(FakeScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []
        self.tp_rank = 0

    def admin(self, action: str, payload: dict):
        self.calls.append((action, payload))
        return {"success": True, "message": "ok", "data": {"action": action}}


def test_admin_messages_round_trip() -> None:
    op = AdminOperation(
        op_id="op-1",
        action="model_info",
        payload={"x": 1},
        target_stages=["decode"],
        timeout_s=12.5,
    )
    msg = AdminMessage(op)

    decoded = deserialize_message(serialize_message(msg))

    assert isinstance(decoded, AdminMessage)
    assert decoded.operation.op_id == "op-1"
    assert decoded.operation.payload == {"x": 1}

    result = AdminResultMessage(
        AdminResult(
            op_id="op-1",
            stage="decode",
            action="model_info",
            success=True,
            data={"model_path": "m"},
        )
    )
    parsed = parse_message(result.to_dict())
    assert isinstance(parsed, AdminResultMessage)
    assert parsed.result.data["model_path"] == "m"


def test_coordinator_admin_waits_for_all_stage_results() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator._running = True
        coordinator.register_stage("decode", "inproc://decode")
        coordinator.register_stage("vocoder", "inproc://vocoder")

        task = asyncio.create_task(
            coordinator.admin("model_info", {"detail": True}, timeout_s=1)
        )
        while len(control_plane.submitted) < 2:
            await asyncio.sleep(0)

        for stage, _, msg in control_plane.submitted:
            assert isinstance(msg, AdminMessage)
            coordinator._handle_admin_result(
                AdminResult(
                    op_id=msg.operation.op_id,
                    stage=stage,
                    action=msg.operation.action,
                    success=True,
                    data={"stage": stage},
                )
            )

        result = await task
        assert result["success"] is True
        assert {item["stage"] for item in result["results"]} == {"decode", "vocoder"}

    asyncio.run(_run())


def test_stage_admin_dispatches_to_scheduler() -> None:
    async def _run() -> None:
        scheduler = AdminScheduler()
        control_plane = RecordingStageControlPlane()
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
        )

        await stage._on_admin(
            AdminMessage(
                AdminOperation(
                    op_id="op-1",
                    action="pause_generation",
                    payload={"mode": "in_place"},
                )
            )
        )

        assert scheduler.calls == [("pause_generation", {"mode": "in_place"})]
        result_msg = control_plane.completions[0]
        assert isinstance(result_msg, AdminResultMessage)
        assert result_msg.result.success is True
        assert result_msg.result.data["action"] == "pause_generation"

    asyncio.run(_run())
