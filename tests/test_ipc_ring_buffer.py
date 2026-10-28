"""Unit tests for IPC messages and shared-memory ring buffer."""

import numpy as np

from fast_float.ipc.messages import IPCFrameNotification, IPCRequest, IPCResponse
from fast_float.ipc.shm_buffer import FrameRingBuffer


def test_ipc_messages_roundtrip():
    req = IPCRequest(
        cmd="infer",
        session_id="sess_123",
        req_id="r1",
        audio_pcm="abc12345",
        valid_samples=32000,
        nfe=6,
    )
    req_json = req.to_json()
    req_back = IPCRequest.from_json(req_json)
    assert req_back.cmd == "infer"
    assert req_back.session_id == "sess_123"
    assert req_back.nfe == 6

    resp = IPCResponse(status="ok", req_id="r1", total_frames=50, rtf=0.65)
    resp_back = IPCResponse.from_json(resp.to_json())
    assert resp_back.status == "ok"
    assert resp_back.total_frames == 50
    assert resp_back.rtf == 0.65

    notif = IPCFrameNotification(
        session_id="sess_123",
        gen=1,
        slot_idx=4,
        frame_idx=10,
        pts_ms=400.0,
        width=512,
        height=512,
        audio_bytes_len=1280,
    )
    notif_back = IPCFrameNotification.from_json(notif.to_json())
    assert notif_back.slot_idx == 4
    assert notif_back.pts_ms == 400.0


def test_shm_ring_buffer_write_read():
    shm_name = "test_shm_float_ring"
    w, h = 512, 512
    ring_write = FrameRingBuffer(name=shm_name, num_slots=8, width=w, height=h, create=True)
    ring_read = FrameRingBuffer(name=shm_name, num_slots=8, width=w, height=h, create=False)

    try:
        # Create test frame (RGB24) and audio
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[100:200, 100:200] = [255, 128, 64]
        audio = b"\x01\x02\x03\x04" * 320  # 1280 bytes

        slot = 3
        ring_write.write_frame(
            slot_idx=slot,
            frame_idx=42,
            pts_ms=1680.0,
            image_array=img,
            audio_bytes=audio,
        )

        # Read from consumer
        f_idx, pts, img_out, audio_out = ring_read.read_frame(slot_idx=slot)
        assert f_idx == 42
        assert pts == 1680.0
        assert np.array_equal(img_out, img)
        assert audio_out == audio
    finally:
        ring_read.close()
        ring_write.close()
        ring_write.unlink()
