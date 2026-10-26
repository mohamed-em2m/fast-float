import React, { useState, useRef, useEffect } from 'react';
import {
  Mic,
  Square,
  Upload,
  Play,
  Pause,
  Volume2,
  RotateCcw,
  FileAudio,
  Music,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import {
  audioFormatFromFile,
  blobToBase64,
  formatBytes,
} from '../services/api';

const SAMPLE_AUDIO = {
  url: '/samples/speech1.wav',
  label: 'Studio sample · English',
};

// Base64 inflates the file by ~33%; keep the JSON request within reason.
const MAX_AUDIO_BYTES = 30 * 1024 * 1024;

// Cap a single microphone take. Long clips make the render take minutes and
// were the most common cause of a "stuck" generation; auto-stop with a toast.
const MAX_RECORD_SECONDS = 60;

const AUDIO_ACCEPT =
  'audio/wav,audio/mpeg,audio/mp4,audio/ogg,audio/webm,audio/flac,.wav,.mp3,.m4a,.mp4,.ogg,.opus,.webm,.flac,.aac';

export default function AudioEngine({
  audioBlob,
  setAudioBlob,
  audioBase64,
  setAudioBase64,
  setAudioFormat,
  step = 2,
  onToast,
}) {
  const [activeTab, setActiveTab] = useState('mic'); // 'mic' | 'upload' | 'samples'
  const [isRecording, setIsRecording] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [sourceLabel, setSourceLabel] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  // null | 'upload' | 'sample'
  const [loading, setLoading] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const streamRef = useRef(null);
  const timerIntervalRef = useRef(null);
  const audioElementRef = useRef(null);
  const fileInputRef = useRef(null);
  const previewUrlRef = useRef(null);
  const startTimeRef = useRef(0);
  const unmountedRef = useRef(false);
  // Latest selection wins: a slow read must not overwrite a newer clip.
  const selectionRef = useRef(0);
  const dragDepthRef = useRef(0);

  const stopTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const clearWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  };

  const releaseCapture = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    stopTracks();
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    mediaRecorderRef.current = null;
  };

  const setPreviewSrc = (blob) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = blob ? URL.createObjectURL(blob) : null;
    const el = audioElementRef.current;
    if (el) {
      el.pause();
      if (previewUrlRef.current) {
        el.src = previewUrlRef.current;
      } else {
        el.removeAttribute('src');
      }
      el.load();
    }
    setIsPlaying(false);
    setPlaybackTime(0);
    setAudioDuration(0);
  };

  // Map a MediaRecorder MIME type to the container ext the backend understands.
  const mimeToFormat = (mime) => audioFormatFromFile({ type: mime }) || 'webm';

  const formatSeconds = (sec) => {
    const safe = Number.isFinite(sec) && sec > 0 ? sec : 0;
    const mins = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60);
    const tenths = Math.floor((safe % 1) * 10);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
  };

  // Clean up audio context, recorder and stream on unmount.
  useEffect(() => {
    const element = audioElementRef.current;
    return () => {
      unmountedRef.current = true;
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        // The element is going away; do not let onstop call setState.
        recorder.onstop = null;
        try {
          recorder.stop();
        } catch {
          /* already stopped */
        }
      }
      releaseCapture();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (element) {
        try {
          element.pause();
          element.removeAttribute('src');
          element.load();
        } catch {
          /* ignore */
        }
      }
    };
    // releaseCapture only touches refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Canvas visualizer loop
  const drawLiveWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    // The gradient only depends on the canvas size; rebuilding it for every
    // bar on every animation frame was needless work at 60 fps.
    const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
    gradient.addColorStop(0, '#6366f1');
    gradient.addColorStop(0.6, '#06b6d4');
    gradient.addColorStop(1, '#f43f5e');
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.2;
      let x = 0;

      ctx.fillStyle = gradient;
      for (let i = 0; i < bufferLength; i += 1) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    };

    render();
  };

  // Start live microphone recording
  const startRecording = async () => {
    if (isRecording || isStarting || loading) return;
    const token = ++selectionRef.current;
    setSelectedSample(null);
    setIsStarting(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      if (unmountedRef.current || token !== selectionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        if (!unmountedRef.current) setIsStarting(false);
        return;
      }
      streamRef.current = stream;

      // Setup Web Audio API Analyzer for live waveform rendering.
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;
      await audioCtx.resume?.().catch(() => {});
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      // An AnalyserNode is only pulled when it reaches the destination. A
      // zero-gain sink keeps the meter running without echoing the mic.
      const analyserSink = audioCtx.createGain();
      analyserSink.gain.value = 0;
      source.connect(analyser);
      analyser.connect(analyserSink);
      analyserSink.connect(audioCtx.destination);

      // Setup MediaRecorder
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
        mimeType = 'audio/ogg;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        mimeType = 'audio/mp4';
      }

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      const chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error', event?.error || event);
        onToast?.('Recording failed. Check that a microphone is available and try again.', 'error');
        setIsRecording(false);
        clearWaveform();
        releaseCapture();
      };

      mediaRecorder.onstop = async () => {
        const recordedBlob = new Blob(chunks, { type: mimeType });
        const duration = (Date.now() - startTimeRef.current) / 1000;
        setIsRecording(false);
        clearWaveform();
        releaseCapture();
        if (unmountedRef.current || token !== selectionRef.current) return;

        if (!recordedBlob.size) {
          onToast?.('No audio was captured. Try recording again.', 'error');
          return;
        }

        setAudioBlob(recordedBlob);
        setSourceLabel(`Microphone take · ${formatSeconds(duration)}`);
        // Send the REAL container format — the backend transcodes it. Labeling
        // webm/opus as 'wav' used to silently produce garbage audio.
        setAudioFormat(mimeToFormat(mimeType));
        setPreviewSrc(recordedBlob);

        try {
          const base64 = await blobToBase64(recordedBlob);
          if (unmountedRef.current || token !== selectionRef.current) return;
          setAudioBase64(base64);
        } catch (err) {
          console.error('Failed to encode recording', err);
          if (token === selectionRef.current) {
            setAudioBase64(null);
            onToast?.('Recorded audio could not be read. Try a shorter take.', 'error');
          }
        }
      };

      startTimeRef.current = Date.now();
      mediaRecorder.start(100);
      setIsRecording(true);
      setIsStarting(false);
      setRecordDuration(0);

      // Start duration timer (4 Hz is enough for a tenths-place readout).
      timerIntervalRef.current = setInterval(() => {
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        setRecordDuration(elapsed);
        if (elapsed >= MAX_RECORD_SECONDS) {
          onToast?.(
            `Recording stopped at ${MAX_RECORD_SECONDS}s. Keep takes short for a fast render.`,
            'info'
          );
          stopRecording();
        }
      }, 250);

      // Start canvas waveform visualizer loop
      drawLiveWaveform();
    } catch (err) {
      console.error('Microphone access denied or failed:', err);
      releaseCapture();
      if (!unmountedRef.current) setIsStarting(false);
      setIsRecording(false);
      const denied = err?.name === 'NotAllowedError' || err?.name === 'SecurityError';
      onToast?.(
        denied
          ? 'Microphone permission is required. Allow it in your browser, or upload an audio file instead.'
          : 'Could not start the microphone. Check your input device, or upload an audio file instead.',
        'error'
      );
    }
  };

  // Stop recording
  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      // Let the recorder emit its final chunk; onstop does the cleanup.
      recorder.stop();
    }
    setIsRecording(false);
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  // Leaving the mic tab while recording would hide the only stop button.
  const switchTab = (next) => {
    if (next !== 'mic' && isRecording) stopRecording();
    setActiveTab(next);
  };

  // File Upload Handler
  const handleFileUpload = async (file) => {
    if (!file) return;
    const format = audioFormatFromFile(file);
    if (!format) {
      onToast?.('Unsupported audio file. Use WAV, MP3, M4A, OGG, WebM or FLAC.', 'error');
      return;
    }
    if (!file.size) {
      onToast?.('That audio file is empty. Choose another clip.', 'error');
      return;
    }
    if (file.size > MAX_AUDIO_BYTES) {
      onToast?.(`That clip is ${formatBytes(file.size)} — the limit is ${formatBytes(MAX_AUDIO_BYTES)}.`, 'error');
      return;
    }

    const token = ++selectionRef.current;
    setLoading('upload');
    try {
      const base64 = await blobToBase64(file);
      if (unmountedRef.current || token !== selectionRef.current) return;
      setAudioBlob(file);
      setAudioBase64(base64);
      setAudioFormat(format);
      setSourceLabel(file.name);
      setSelectedSample(null);
      setPreviewSrc(file);
    } catch (err) {
      console.error('Failed to read audio file', err);
      if (token === selectionRef.current) {
        onToast?.('Could not read that audio file. Try another clip.', 'error');
      }
    } finally {
      if (token === selectionRef.current) setLoading(null);
    }
  };

  // Select Sample Audio Handler
  const handleSelectSample = async () => {
    const token = ++selectionRef.current;
    setLoading('sample');
    try {
      const res = await fetch(SAMPLE_AUDIO.url, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`Sample fetch failed: ${res.status}`);
      const blob = await res.blob();
      if (!blob.size) throw new Error('empty sample');
      if (blob.size > MAX_AUDIO_BYTES) throw new Error('sample too large');
      const format = audioFormatFromFile(blob) || audioFormatFromFile({ name: SAMPLE_AUDIO.url }) || 'wav';
      const base64 = await blobToBase64(blob);
      if (unmountedRef.current || token !== selectionRef.current) return;
      setAudioBlob(blob);
      setAudioBase64(base64);
      setAudioFormat(format);
      setSourceLabel(SAMPLE_AUDIO.label);
      setSelectedSample(SAMPLE_AUDIO.url);
      setPreviewSrc(blob);
      onToast?.('Sample clip loaded — press Generate when ready.', 'success');
    } catch (err) {
      console.error('Failed to load sample audio', err);
      if (token === selectionRef.current) {
        onToast?.('Could not load the sample clip. Check that the sample audio is available.', 'error');
      }
    } finally {
      if (token === selectionRef.current) setLoading(null);
    }
  };

  // Play / Pause preview audio
  const togglePlayAudio = async () => {
    const el = audioElementRef.current;
    if (!el || !el.getAttribute('src')) return;
    if (!el.paused) {
      el.pause();
      return;
    }
    try {
      // Replay from the start once the clip has ended.
      if (el.ended || (Number.isFinite(el.duration) && el.currentTime >= el.duration)) {
        el.currentTime = 0;
      }
      await el.play();
    } catch (err) {
      console.error('Audio preview failed', err);
      onToast?.('Playback was blocked or this audio format is not supported by your browser.', 'error');
    }
  };

  const handleAudioTimeUpdate = () => {
    if (audioElementRef.current) {
      setPlaybackTime(audioElementRef.current.currentTime);
      const duration = audioElementRef.current.duration;
      if (Number.isFinite(duration) && duration > 0) setAudioDuration(duration);
    }
  };

  const handleSeek = (event) => {
    const el = audioElementRef.current;
    const duration = el?.duration;
    if (!el || !Number.isFinite(duration) || duration <= 0) return;
    const box = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    el.currentTime = ratio * duration;
    setPlaybackTime(el.currentTime);
  };

  const handleClearAudio = () => {
    selectionRef.current += 1;
    setLoading(null);
    setAudioBlob(null);
    setAudioBase64(null);
    setAudioFormat('wav');
    setIsPlaying(false);
    setSourceLabel('');
    setSelectedSample(null);
    setRecordDuration(0);
    setPreviewSrc(null);
    setActiveTab('mic');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="card-section glass-panel">
      {/* Hidden audio element for preview playback */}
      <audio
        ref={audioElementRef}
        preload="metadata"
        onTimeUpdate={handleAudioTimeUpdate}
        onLoadedMetadata={handleAudioTimeUpdate}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        onError={() => {
          const el = audioElementRef.current;
          if (!el || !el.getAttribute('src')) return;
          onToast?.('This browser cannot play the selected audio. Try WAV or MP3.', 'error');
        }}
      />

      <div className="section-header">
        <div className="section-title-wrap">
          <span className="step-index">{step}</span>
          <Volume2 size={18} className="section-icon text-cyan" />
          <h2 className="section-title">Voice</h2>
        </div>
        {audioBase64 && (
          <span className="badge-ready">
            <CheckCircle2 size={13} />
            <span>Audio ready</span>
          </span>
        )}
      </div>

      {/* Tabs */}
      <div className="audio-tab-nav">
        <button
          className={`tab-btn ${activeTab === 'mic' ? 'active' : ''}`}
          onClick={() => switchTab('mic')}
        >
          <Mic size={15} />
          <span>Microphone</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => switchTab('upload')}
        >
          <Upload size={15} />
          <span>Upload File</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'samples' ? 'active' : ''}`}
          onClick={() => switchTab('samples')}
        >
          <Music size={15} />
          <span>Sample Speech</span>
        </button>
      </div>

      {/* TAB 1: LIVE MICROPHONE RECORDING */}
      {activeTab === 'mic' && (
        <div className="mic-record-panel">
          {/* Canvas Waveform Display */}
          <div className="waveform-canvas-box">
            <canvas
              ref={canvasRef}
              width={420}
              height={90}
              className="waveform-canvas"
              aria-hidden="true"
            />
            {!isRecording && !audioBlob && (
              <div className="waveform-idle-overlay">
                <p>Tap record and speak naturally</p>
                <span>16 kHz mono · up to 60 s recommended</span>
              </div>
            )}
          </div>

          {/* Controls Bar */}
          <div className="record-controls-row">
            {!isRecording ? (
              <button
                type="button"
                className="btn-record"
                onClick={startRecording}
                disabled={Boolean(loading) || isStarting}
              >
                {loading || isStarting ? (
                  <Loader2 size={16} className="spin" />
                ) : (
                  <div className="record-dot" />
                )}
                <span>{isStarting ? 'Starting…' : 'Start Recording'}</span>
              </button>
            ) : (
              <button
                type="button"
                className="btn-stop-record recording-pulse"
                onClick={stopRecording}
              >
                <Square size={16} fill="white" />
                <span>Stop ({formatSeconds(recordDuration)})</span>
              </button>
            )}

            {isRecording && (
              <span className="live-rec-badge">
                <span className="rec-blip" />
                <span>REC {formatSeconds(recordDuration)}</span>
              </span>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: UPLOAD AUDIO */}
      {activeTab === 'upload' && (
        <div
          className={`audio-dropzone ${isDragging ? 'is-dragging' : ''}`}
          onClick={() => !loading && fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragEnter={() => {
            dragDepthRef.current += 1;
            setIsDragging(true);
          }}
          onDragLeave={() => {
            dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
            if (!dragDepthRef.current) setIsDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            dragDepthRef.current = 0;
            setIsDragging(false);
            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (file) handleFileUpload(file);
          }}
          aria-busy={loading === 'upload'}
        >
          <input
            type="file"
            ref={fileInputRef}
            accept={AUDIO_ACCEPT}
            onChange={(e) => {
              const file = e.target.files?.[0];
              // Reset first so choosing the same file twice still fires.
              e.target.value = '';
              handleFileUpload(file);
            }}
            style={{ display: 'none' }}
          />
          {loading === 'upload' ? (
            <Loader2 size={28} className="upload-audio-icon spin" />
          ) : (
            <FileAudio size={28} className="upload-audio-icon" />
          )}
          <p className="upload-hint-main">
            {loading === 'upload' ? 'Reading audio…' : 'Click or drop an audio file here'}
          </p>
          <p className="upload-hint-sub">WAV, MP3, M4A, OGG, WebM, FLAC · 16 kHz mono works best</p>
        </div>
      )}

      {/* TAB 3: SAMPLES */}
      {activeTab === 'samples' && (
        <div className="sample-audio-list">
          <div className={`sample-audio-card ${selectedSample ? 'is-selected' : ''}`}>
            <div className="sample-audio-info">
              <span className="sample-audio-name">{SAMPLE_AUDIO.label}</span>
              <span className="sample-audio-desc">A short natural speech clip to try the pipeline</span>
            </div>
            <button
              type="button"
              className="btn-secondary btn-sm"
              onClick={handleSelectSample}
              disabled={loading === 'sample'}
            >
              {loading === 'sample' ? <Loader2 size={13} className="spin" /> : <Play size={13} />}
              {loading === 'sample' ? 'Loading…' : selectedSample ? 'Reload sample' : 'Use sample'}
            </button>
          </div>
        </div>
      )}

      {/* PREVIEW PLAYER ROW (if audio is available) */}
      {audioBase64 && (
        <div className="audio-preview-row">
          <button
            type="button"
            className="btn-play-toggle"
            onClick={togglePlayAudio}
            aria-label={isPlaying ? 'Pause audio preview' : 'Play audio preview'}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <div className="audio-timeline-wrap">
            <div className="audio-preview-meta">
              <span className="audio-source-label">{sourceLabel || 'Audio ready'}</span>
              <span>{formatSeconds(playbackTime)} / {formatSeconds(audioDuration || recordDuration)}</span>
            </div>
            <div
              className="audio-progress-bar"
              onClick={handleSeek}
              role="progressbar"
              aria-label="Audio preview position"
              aria-valuemin={0}
              aria-valuemax={Math.round(audioDuration || 0)}
              aria-valuenow={Math.round(playbackTime || 0)}
              title="Click to seek"
            >
              <div
                className="audio-progress-fill"
                style={{
                  width: `${audioDuration > 0 ? Math.min(100, (playbackTime / audioDuration) * 100) : 0}%`,
                }}
              />
            </div>
          </div>
          <button
            type="button"
            className="btn-icon-subtle"
            title="Clear audio"
            aria-label="Clear audio"
            onClick={handleClearAudio}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
