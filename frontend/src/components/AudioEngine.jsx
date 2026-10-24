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
  CheckCircle2
} from 'lucide-react';
import { blobToBase64 } from '../services/api';

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
  const [recordDuration, setRecordDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [sourceLabel, setSourceLabel] = useState('');
  const [isDragging, setIsDragging] = useState(false);

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

  const stopTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const setPreviewSrc = (blob) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = blob ? URL.createObjectURL(blob) : null;
    if (audioElementRef.current) audioElementRef.current.src = previewUrlRef.current || '';
  };

  // Map a MediaRecorder MIME type to the container ext the backend understands.
  const mimeToFormat = (mime) => {
    if (mime.includes('mp4')) return 'mp4';
    if (mime.includes('ogg')) return 'ogg';
    return 'webm';
  };

  // Clean up audio context and stream on unmount
  useEffect(() => {
    return () => {
      stopTracks();
      if (audioContextRef.current) audioContextRef.current.close().catch(() => {});
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  // Start live microphone recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true
        }
      });
      streamRef.current = stream;

      // Setup Web Audio API Analyzer for live waveform rendering
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

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
        if (e.data && e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const recordedBlob = new Blob(chunks, { type: mimeType });
        setAudioBlob(recordedBlob);
        setSourceLabel(`Microphone take · ${formatSeconds((Date.now() - startTime) / 1000)}`);
        // Send the REAL container format — the backend transcodes it.
        // Labeling webm/opus as 'wav' used to silently produce garbage audio.
        setAudioFormat(mimeToFormat(mimeType));

        const base64 = await blobToBase64(recordedBlob);
        setAudioBase64(base64);

        setPreviewSrc(recordedBlob);
        stopTracks();
        if (audioContextRef.current) {
          audioContextRef.current.close().catch(() => {});
          audioContextRef.current = null;
        }
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordDuration(0);

      // Start duration timer
      const startTime = Date.now();
      timerIntervalRef.current = setInterval(() => {
        setRecordDuration((Date.now() - startTime) / 1000);
      }, 100);

      // Start canvas waveform visualizer loop
      drawLiveWaveform();
    } catch (err) {
      console.error('Microphone access denied or failed:', err);
      onToast?.(
        'Microphone permission is required. Allow it in your browser, or upload an audio file instead.',
        'error'
      );
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
  };

  // Canvas visualizer loop
  const drawLiveWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.2;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;

        // Gradient for waveform
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(0.6, '#06b6d4');
        gradient.addColorStop(1, '#f43f5e');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);

        x += barWidth;
      }
    };

    render();
  };

  // File Upload Handler
  const handleFileUpload = async (file) => {
    if (!file) return;
    setAudioBlob(file);
    setSourceLabel(file.name);
    // Prefer the real MIME container over the filename; fall back to ext.
    const mimeExt = file.type.includes('mp4') || file.type.includes('m4a')
      ? 'mp4'
      : file.type.includes('mpeg')
        ? 'mp3'
        : file.type.includes('ogg')
          ? 'ogg'
          : file.type.includes('webm')
            ? 'webm'
            : file.type.includes('wav') || file.type.includes('wave')
              ? 'wav'
              : (file.name.split('.').pop().toLowerCase() || 'wav');
    setAudioFormat(mimeExt);

    const base64 = await blobToBase64(file);
    setAudioBase64(base64);

    setPreviewSrc(file);
  };

  // Select Sample Audio Handler
  const handleSelectSample = async (url) => {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Sample fetch failed: ${res.status}`);
      const blob = await res.blob();
      setAudioBlob(blob);
      setSourceLabel('Sample speech clip');
      setAudioFormat(url.split('.').pop().toLowerCase() || 'wav');
      const base64 = await blobToBase64(blob);
      setAudioBase64(base64);

      setPreviewSrc(blob);
    } catch (e) {
      console.error('Failed to load sample audio', e);
    }
  };

  // Play / Pause preview audio
  const togglePlayAudio = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    } else {
      audioElementRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleAudioTimeUpdate = () => {
    if (audioElementRef.current) {
      setPlaybackTime(audioElementRef.current.currentTime);
      setAudioDuration(audioElementRef.current.duration || 0);
    }
  };

  const formatSeconds = (sec) => {
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    const tenths = Math.floor((sec % 1) * 10);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
  };

  return (
    <div className="card-section glass-panel">
      {/* Hidden audio element for preview playback */}
      <audio
        ref={audioElementRef}
        onTimeUpdate={handleAudioTimeUpdate}
        onEnded={() => setIsPlaying(false)}
        onLoadedMetadata={handleAudioTimeUpdate}
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
          onClick={() => setActiveTab('mic')}
        >
          <Mic size={15} />
          <span>Microphone</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          <Upload size={15} />
          <span>Upload File</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'samples' ? 'active' : ''}`}
          onClick={() => setActiveTab('samples')}
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
              >
                <div className="record-dot" />
                <span>Start Recording</span>
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
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
          }}
        >
          <input
            type="file"
            ref={fileInputRef}
            accept="audio/*"
            onChange={(e) => handleFileUpload(e.target.files[0])}
            style={{ display: 'none' }}
          />
          <FileAudio size={28} className="upload-audio-icon" />
          <p className="upload-hint-main">Click or drop an audio file here</p>
          <p className="upload-hint-sub">WAV, MP3, M4A, OGG · 16 kHz mono works best</p>
        </div>
      )}

      {/* TAB 3: SAMPLES */}
      {activeTab === 'samples' && (
        <div className="sample-audio-list">
          <div className="sample-audio-card">
            <div className="sample-audio-info">
              <span className="sample-audio-name">Studio sample · English</span>
              <span className="sample-audio-desc">A short natural speech clip to try the pipeline</span>
            </div>
            <button
              type="button"
              className="btn-secondary btn-sm"
              onClick={() => handleSelectSample('/samples/speech1.wav')}
            >
              <Play size={13} />
              Use sample
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
            <div className="audio-progress-bar">
              <div
                className="audio-progress-fill"
                style={{
                  width: `${audioDuration > 0 ? (playbackTime / audioDuration) * 100 : 0}%`
                }}
              />
            </div>
          </div>
          <button
            type="button"
            className="btn-icon-subtle"
            title="Clear audio"
            aria-label="Clear audio"
            onClick={() => {
              setAudioBlob(null);
              setAudioBase64(null);
              setIsPlaying(false);
              setSourceLabel('');
              setPreviewSrc(null);
            }}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
