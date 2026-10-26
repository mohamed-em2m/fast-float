"""Unit tests for ZeroMQ IPC worker message serialization and handling."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from fast_float.ipc.messages import IPCFrameNotification, IPCRequest, IPCResponse
from fast_float.ipc.shm_buffer import FrameRingBuffer
from fast_float.workers.inference_worker import (
    _Cancelled,
    _ShmFrameDispatcher,
)


def test_ipc_messages_serialization():
    req = IPCRequest(
        cmd="infer",
        session_id="sess-123",
        req_id="req-456",
        audio_pcm="dGVzdA==",
        valid_samples=16000,
        sample_rate=16000,
        nfe=5,
        ref_image_path="test.png",
        gen=1,
    )
    req_json = req.to_json()
    req_recovered = IPCRequest.from_json(req_json)
    assert req_recovered.cmd == "infer"
    assert req_recovered.session_id == "sess-123"
    assert req_recovered.nfe == 5

    resp = IPCResponse(
        status="ok",
        req_id="req-456",
        session_id="sess-123",
        total_frames=50,
        rtf=0.85,
    )
    resp_json = resp.to_json()
    resp_recovered = IPCResponse.from_json(resp_json)
    assert resp_recovered.status == "ok"
    assert resp_recovered.total_frames == 50
    assert resp_recovered.rtf == 0.85

    notif = IPCFrameNotification(
        session_id="sess-123",
        gen=1,
        slot_idx=3,
        frame_idx=10,
        pts_ms=400.0,
        width=512,
        height=512,
        audio_bytes_len=1280,
        is_last=False,
    )
    notif_json = notif.to_json()
    notif_recovered = IPCFrameNotification.from_json(notif_json)
    assert notif_recovered.frame_idx == 10
    assert notif_recovered.pts_ms == 400.0
    assert not notif_recovered.is_last


def test_shm_frame_dispatcher():
    import threading

    ring = FrameRingBuffer("test_ring_disp", num_slots=8, create=True)
    mock_pub = MagicMock()
    interrupt = threading.Event()

    audio = np.zeros(16000, dtype=np.int16)
    dispatcher = _ShmFrameDispatcher(
        ring=ring,
        pub_socket=mock_pub,
        session_id="sess-test",
        req_id="req-test",
        gen=1,
        interrupt_event=interrupt,
        audio_pcm_int16=audio,
    )

    fake_frames = np.zeros((2, 512, 512, 3), dtype=np.uint8)
    dispatcher.dispatch(idx=0, frames=fake_frames, pts_ms=0.0)

    # Verify PUB socket sent 2 frame notifications
    assert mock_pub.send_string.call_count == 2
    sent_notif = IPCFrameNotification.from_json(mock_pub.send_string.call_args_list[0][0][0])
    assert sent_notif.frame_idx == 0
    assert sent_notif.session_id == "sess-test"
    assert sent_notif.slot_idx == 0

    # Verify frame is in ring buffer
    frame_idx, _pts, arr, a_bytes = ring.read_frame(0)
    assert frame_idx == 0
    assert arr.shape == (512, 512, 3)
    assert len(a_bytes) == 1280

    # Test interrupt raising _Cancelled
    interrupt.set()
    with pytest.raises(_Cancelled):
        dispatcher.dispatch(idx=2, frames=fake_frames, pts_ms=80.0)

    # Test close sends sentinel notification
    dispatcher.close()
    last_notif = IPCFrameNotification.from_json(mock_pub.send_string.call_args_list[-1][0][0])
    assert last_notif.is_last is True

    ring.close()
    ring.unlink()
