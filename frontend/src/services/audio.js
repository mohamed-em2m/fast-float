/**
 * Live microphone capture for the avatar session.
 *
 * Browsers give us 44.1/48 kHz float samples. FLOAT wants mono int16 at
 * 16 kHz, so each ~200 ms batch is downsampled (box average, which also
 * low-passes), converted to PCM16 and posted by `LiveAudioStudio`.
 */

import { pushConversationAudio } from './livekit';

const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 3200; // 200 ms at 16 kHz

/** Box-filter a float buffer down to 16 kHz. */
export function downsampleTo16k(input, inputRate) {
  if (inputRate === TARGET_RATE) return new Float32Array(input);
  const ratio = inputRate / TARGET_RATE;
  const length = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    output[i] = end > start ? sum / (end - start) : 0;
  }
  return output;
}

function floatToPcm16(input) {
  const pcm = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm;
}

function concatPcm16(chunks, totalSamples) {
  const output = new Int16Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

/** Base64-encode an Int16Array without blowing the call stack on large chunks. */
export function pcm16ToBase64(pcm) {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  let binary = '';
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
  }
  return btoa(binary);
}

/**
 * Capture the microphone and stream it to a running conversation session.
 *
 * Call `stop()` when the session ends; it flushes the tail so the last words
 * are still rendered. Media errors are reported through `onError`.
 */
export class MicStreamer {
  constructor({ room, onError, onLevel } = {}) {
    this.room = room;
    this.onError = onError;
    this.onLevel = onLevel;
    this.stream = null;
    this.context = null;
    this.processor = null;
    this.analyser = null;
    this.source = null;
    this.sink = null;
    this.buffer = [];
    this.bufferedSamples = 0;
    this.muted = false;
    this.stopped = false;
    this.sendChain = Promise.resolve();
    this.analyserData = null;
    this.levelTimer = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.context = new AudioCtx();
    // A user gesture started the session; this is a no-op when already running.
    await this.context.resume?.().catch(() => {});

    this.source = this.context.createMediaStreamSource(this.stream);
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.75;
    this.analyserData = new Uint8Array(this.analyser.frequencyBinCount);
    this.source.connect(this.analyser);

    // ScriptProcessor is deprecated but universally available and good enough
    // for 200 ms PCM batches (this is not an audio-quality path).
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.sink = this.context.createGain();
    this.sink.gain.value = 0; // keep the graph pulled without echoing to speakers

    this.processor.onaudioprocess = (event) => {
      if (this.stopped || this.muted) return;
      const channel = event.inputBuffer.getChannelData(0);
      const pcm = floatToPcm16(downsampleTo16k(channel, this.context.sampleRate));
      this.buffer.push(pcm);
      this.bufferedSamples += pcm.length;
      if (this.bufferedSamples >= CHUNK_SAMPLES) this._send();
    };

    this.source.connect(this.processor);
    this.processor.connect(this.sink);
    this.sink.connect(this.context.destination);

    if (this.onLevel) {
      this.levelTimer = setInterval(() => this._emitLevel(), 100);
    }
  }

  _emitLevel() {
    if (!this.analyser || !this.analyserData || !this.onLevel) return;
    this.analyser.getByteTimeDomainData(this.analyserData);
    let peak = 0;
    for (let i = 0; i < this.analyserData.length; i += 1) {
      peak = Math.max(peak, Math.abs(this.analyserData[i] - 128) / 128);
    }
    this.onLevel(this.muted ? 0 : peak);
  }

  _send() {
    if (!this.bufferedSamples) return;
    const pcm = concatPcm16(this.buffer, this.bufferedSamples);
    this.buffer = [];
    this.bufferedSamples = 0;
    const base64 = pcm16ToBase64(pcm);
    // Serialise posts: the renderer owns the ordering, and a slow network must
    // not let two chunks race.
    this.sendChain = this.sendChain
      .then(() => pushConversationAudio(this.room, base64))
      .catch((err) => this.onError?.(err.message || 'Live audio upload failed'));
  }

  setMuted(muted) {
    this.muted = muted;
    // Drop whatever was captured before the mute so it is not sent afterwards.
    if (muted) {
      this.buffer = [];
      this.bufferedSamples = 0;
    }
  }

  async stop({ flush = true } = {}) {
    if (this.stopped) return;
    this.stopped = true;
    if (this.levelTimer) clearInterval(this.levelTimer);
    if (this.processor) {
      this.processor.onaudioprocess = null;
      try {
        this.processor.disconnect();
      } catch {
        /* ignore */
      }
    }
    try {
      this.source?.disconnect();
      this.sink?.disconnect();
    } catch {
      /* ignore */
    }
    if (this.bufferedSamples) this._send();
    await this.sendChain.catch(() => {});
    if (flush) {
      await pushConversationAudio(this.room, '', { flush: true }).catch(() => {});
    }
    try {
      this.context?.close();
    } catch {
      /* ignore */
    }
    try {
      this.stream?.getTracks().forEach((track) => track.stop());
    } catch {
      /* ignore */
    }
    this.context = null;
    this.processor = null;
    this.analyser = null;
    this.stream = null;
  }
}
