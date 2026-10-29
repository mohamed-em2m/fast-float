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


def test_ring_flow_control_waits_for_consumer():
    """A slot that has not been read must not be overwritten.

    The ring is round-robin, so a writer that outruns the reader silently
    replaces unread frames: the consumer then plays a *newer* frame's picture
    and audio at an older index, which is heard as the avatar jumping forward
    mid-sentence (and, because each chunk restarts the slot counter at 0, as
    the next chunk's audio landing on the previous chunk's unread tail).
    """
    import threading
    import time

    shm_name = "test_shm_float_flow"
    w, h = 64, 64
    ring_write = FrameRingBuffer(name=shm_name, num_slots=4, width=w, height=h, create=True)
    ring_read = FrameRingBuffer(name=shm_name, num_slots=4, width=w, height=h, create=False)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    try:
        ring_write.write_frame(0, 0, 0.0, img, b"\x01\x02")
        # Slot 0 is busy: a write with a timeout must refuse (and report it)
        # rather than clobbering the unread frame.
        assert ring_write.write_frame(0, 99, 0.0, img, b"", wait_s=0.05) is False
        assert ring_write.forced_overwrites == 1

        # Once the consumer has taken that frame the slot is free again.
        assert ring_read.read_frame(0)[0] == 99
        assert ring_write.write_frame(0, 7, 40.0, img, b"\x03\x04", wait_s=1.0) is True
        idx, _pts, _arr, audio = ring_read.read_frame(0)
        assert idx == 7
        assert audio == b"\x03\x04"
        assert ring_write.write_frame(0, 8, 80.0, img, b"", wait_s=0.05) is True

        # A consumer that releases a skipped slot unblocks a waiting writer.
        ring_write.write_frame(1, 1, 40.0, img, b"")
        done = threading.Event()

        def _write():
            ring_write.write_frame(1, 2, 80.0, img, b"", wait_s=2.0)
            done.set()

        thread = threading.Thread(target=_write, daemon=True)
        thread.start()
        time.sleep(0.1)
        assert not done.is_set()  # blocked on the unread slot
        ring_read.release_slot(1)
        assert done.wait(1.0)
        thread.join(1.0)
    finally:
        ring_read.close()
        ring_write.close()
        ring_write.unlink()
