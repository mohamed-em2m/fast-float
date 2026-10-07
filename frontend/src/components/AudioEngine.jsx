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
  audioFormat, 
  setAudioFormat 
}) {
  const [activeTab, setActiveTab] = useState('mic'); // 'mic' | 'upload' | 'samples'
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const streamRef = useRef(null);
  const timerIntervalRef = useRef(null);
  const audioElementRef = useRef(null);
  const fileInputRef = useRef(null);

  // Clean up audio context and stream on unmount
  useEffect(() => {
    return () => {
      stopTracks();
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  const stopTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

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
        setAudioFormat('wav'); // FastAPI converts webm/opus/raw seamlessly

        const base64 = await blobToBase64(recordedBlob);
        setAudioBase64(base64);

        if (audioElementRef.current) {
          audioElementRef.current.src = URL.createObjectURL(recordedBlob);
        }
        stopTracks();
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
      alert('Microphone permission is required to record speech.');
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
    const ext = file.name.split('.').pop().toLowerCase() || 'wav';
    setAudioFormat(ext);

    const base64 = await blobToBase64(file);
    setAudioBase64(base64);

    if (audioElementRef.current) {
      audioElementRef.current.src = URL.createObjectURL(file);
    }
  };

  // Select Sample Audio Handler
  const handleSelectSample = async (url) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      setAudioBlob(blob);
      setAudioFormat('wav');
      const base64 = await blobToBase64(blob);
      setAudioBase64(base64);

      if (audioElementRef.current) {
        audioElementRef.current.src = URL.createObjectURL(blob);
      }
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
          <Volume2 size={18} className="section-icon text-cyan" />
          <h2 className="section-title">Driving Speech Audio</h2>
        </div>
        {audioBase64 && (
          <span className="badge-ready">
            <CheckCircle2 size={13} />
            <span>Audio Ready</span>
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
            />
            {!isRecording && !audioBlob && (
              <div className="waveform-idle-overlay">
                <p>Press Record and speak clearly into your mic</p>
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
          className="audio-dropzone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
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
          <p className="upload-hint-main">Click or drop audio file here</p>
          <p className="upload-hint-sub">WAV, MP3, AAC, PCM (16kHz recommended)</p>
        </div>
      )}

      {/* TAB 3: SAMPLES */}
      {activeTab === 'samples' && (
        <div className="sample-audio-list">
          <div className="sample-audio-card">
            <div className="sample-audio-info">
              <span className="sample-audio-name">Sample Speech Clip 1</span>
              <span className="sample-audio-desc">Natural English conversation clip (0.5s - 4s)</span>
            </div>
            <button 
              type="button" 
              className="btn-secondary btn-sm"
              onClick={() => handleSelectSample('/samples/speech1.wav')}
            >
              Use Sample
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
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <div className="audio-timeline-wrap">
            <div className="audio-progress-bar">
              <div 
                className="audio-progress-fill" 
                style={{ 
                  width: `${audioDuration > 0 ? (playbackTime / audioDuration) * 100 : 0}%` 
                }} 
              />
            </div>
            <div className="audio-time-labels">
              <span>{formatSeconds(playbackTime)}</span>
              <span>{formatSeconds(audioDuration || recordDuration)}</span>
            </div>
          </div>
          <button 
            type="button" 
            className="btn-icon-subtle" 
            title="Re-record / Clear"
            onClick={() => {
              setAudioBlob(null);
              setAudioBase64(null);
              if (audioElementRef.current) audioElementRef.current.src = '';
            }}
          >
            <RotateCcw size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
