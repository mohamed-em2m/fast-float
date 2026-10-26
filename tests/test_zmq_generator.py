"""Tests for ZmqFloatVideoGenerator."""

from unittest.mock import AsyncMock, patch

import pytest

from fast_float.livekit.zmq_generator import ZmqFloatVideoGenerator


@pytest.mark.asyncio
async def test_zmq_generator_initialization():
    gen = ZmqFloatVideoGenerator(
        ref_image_path="ref.png",
        nfe=5,
        chunk_sec=2.0,
        first_chunk_sec=2.0,
        shm_name="test_shm_init",
        shm_slots=8,
    )
    assert gen._nfe == 5
    assert gen._chunk_sec == 2.0
    assert gen._audio_samples == 0


@pytest.mark.asyncio
async def test_zmq_generator_push_audio_and_clear():
    gen = ZmqFloatVideoGenerator(
        ref_image_path="ref.png",
        nfe=5,
        chunk_sec=2.0,
        first_chunk_sec=2.0,
        shm_name="test_shm_push",
        shm_slots=8,
    )

    # Mock _dispatch_infer so it does not send actual network requests
    with patch.object(gen, "_dispatch_infer", new_callable=AsyncMock) as mock_dispatch:
        # Push 1 second of 16kHz audio (16000 samples = 32000 bytes)
        one_sec_pcm = b"\x00" * 32000
        gen.push_audio(one_sec_pcm)
        assert gen._audio_samples == 16000
        assert mock_dispatch.call_count == 0

        # Push another 1 second to reach chunk threshold (32000 samples)
        gen.push_audio(one_sec_pcm)
        # Buffer should be drained and dispatched
        assert gen._audio_samples == 0
        assert gen._first_chunk_done is True
        assert gen._gen == 1

    # Test clear_buffer
    with patch.object(gen, "_send_interrupt", new_callable=AsyncMock) as mock_interrupt:
        gen.push_audio(b"\x00" * 1000)
        assert gen._audio_samples > 0
        gen.clear_buffer()
        assert gen._audio_samples == 0
        assert gen._first_chunk_done is False
