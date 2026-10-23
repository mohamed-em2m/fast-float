import React, { useCallback, useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Download,
  Share2,
  Radio,
  Film,
  Zap,
  Gauge
} from 'lucide-react';
import { attachRemoteTrack, createWebRtcSession, getApiUrl, setIceServers } from '../services/api';
import { connectLiveKitSession, fetchLiveKitStatus, triggerLiveKitPublish } from '../services/livekit';
import LivePreview from './LivePreview';

const GENERATION_STAGES = [
  { label: 'Preprocessing Audio & Image', icon: '🎧' },
  { label: 'Encoding Motion Latent Flow', icon: '⚡' },
  { label: 'Solving Transformer ODE Trajectory', icon: '🧠' },
  { label: 'NVENC Hardware Frame Compression', icon: '🎬' },
];

const TRANSPORTS = [
  { id: 'mp4', label: 'MP4', title: 'Render the whole video, then play the file' },
  { id: 'livekit', label: 'LiveKit', title: 'Watch frames live over LiveKit SFU (WebRTC)' },
  { id: 'webrtc', label: 'WebRTC', title: 'Watch frames live over WebRTC (aiortc)' },
  { id: 'ws', label: 'WebSocket', title: 'Watch frames live over a WebSocket' },
];

export default function StudioStage({
  onGenerate,
  isGenerating,
  progress,
  generationResult,
  hasAudio,
  onToast,
  transport = 'mp4',
  setTransport,
  liveRequestId,
}) {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [useWebRtc, setUseWebRtc] = useState(false);
  const [webRtcConnected, setWebRtcConnected] = useState(false);
  const [useLiveKit, setUseLiveKit] = useState(false);
  const [liveKitConnected, setLiveKitConnected] = useState(false);
  // null = unknown (checking), true/false = server responded
  const [liveKitAvailable, setLiveKitAvailable] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const videoRef = useRef(null);
  const webRtcVideoRef = useRef(null);
  const pcRef = useRef(null);
  const liveKitSessionRef = useRef(null);

  // Check LiveKit availability on mount and whenever the active API URL changes.
  useEffect(() => {
    const checkLiveKit = () => {
      fetchLiveKitStatus()
        .then((status) => {
          setLiveKitAvailable(!!status.configured);
          // Feed the deployment's own STUN list (self-hosted coturn or cloud)
          // to the raw WebRTC transport; empty keeps the Google default.
          setIceServers(status.stun_urls);
        })
        .catch(() => setLiveKitAvailable(false));
    };
    checkLiveKit();
    window.addEventListener('float:api_url_changed', checkLiveKit);
    return () => window.removeEventListener('float:api_url_changed', checkLiveKit);
  }, []);

  // Automatic transport fallback: LiveKit is the primary SFU. When it is
  // unconfigured or unreachable, switch to the raw WebRTC transport whose
  // signalling and STUN live on this API host — not to the WebSocket
  // transport, which is only the last resort (LiveKit → WebRTC → WebSocket).
  useEffect(() => {
    if (liveKitAvailable === false && transport === 'livekit') {
      setTransport?.('webrtc');
      if (onToast) onToast('LiveKit unavailable — switched to WebRTC via the main server', 'error');
    }
  }, [liveKitAvailable, transport, setTransport, onToast]);

  const handleLiveFallback = useCallback(() => {
    const next = transport === 'livekit' ? 'webrtc' : 'ws';
    setTransport?.(next);
    if (onToast) {
      onToast(
        next === 'webrtc'
          ? 'LiveKit SFU failed — switched to WebRTC via the main server'
          : 'WebRTC failed — switched to the WebSocket transport',
        'error'
      );
    }
  }, [transport, setTransport, onToast]);

  // Cycle progress stages smoothly during generation
  useEffect(() => {
    let interval;
    if (isGenerating) {
      setCurrentStageIndex(0);
      interval = setInterval(() => {
        setCurrentStageIndex((prev) => (prev < GENERATION_STAGES.length - 1 ? prev + 1 : prev));
      }, 1400);
    } else {
      setCurrentStageIndex(0);
    }
    return () => clearInterval(interval);
  }, [isGenerating]);

  const stopTracks = (stream) => {
    try {
      (stream?.getTracks() || []).forEach((t) => t.stop());
    } catch {
      /* ignore */
    }
  };

  const handleStopWebRtc = () => {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    stopTracks(webRtcVideoRef.current?.srcObject);
    if (webRtcVideoRef.current) webRtcVideoRef.current.srcObject = null;
    setUseWebRtc(false);
    setWebRtcConnected(false);
  };

  const handleStopLiveKit = () => {
    // livekit-client manages its own media elements; calling disconnect() on
    // the session object (returned by connectLiveKitSession) is sufficient.
    if (liveKitSessionRef.current) {
      liveKitSessionRef.current.disconnect();
      liveKitSessionRef.current = null;
    }
    setUseLiveKit(false);
    setLiveKitConnected(false);
  };

  // Clean up WebRTC peer connection and LiveKit session
  useEffect(() => {
    return () => {
      try {
        (webRtcVideoRef.current?.srcObject?.getTracks() || []).forEach((t) => t.stop());
      } catch {
        /* ignore */
      }
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
      if (liveKitSessionRef.current) {
        liveKitSessionRef.current.disconnect();
        liveKitSessionRef.current = null;
      }
    };
  }, []);

  const handleStartLiveKit = async () => {
    if (!generationResult?.request_id) return;
    try {
      handleStopWebRtc();
      setUseLiveKit(true);
      setLiveKitConnected(false);

      // Trigger publishing of completed file or stream
      await triggerLiveKitPublish(generationResult.request_id);

      const session = await connectLiveKitSession({
        room: generationResult.request_id,
        videoElement: webRtcVideoRef.current,
        onStatus: (state) => {
          if (state === 'live') setLiveKitConnected(true);
          if (state === 'failed' || state === 'error') {
            setLiveKitConnected(false);
            if (onToast) onToast('LiveKit stream failed — back to MP4', 'error');
            handleStopLiveKit();
          }
          if (state === 'done') setLiveKitConnected(false);
        },
      });
      liveKitSessionRef.current = session;
      if (onToast) onToast('LiveKit stream established!', 'success');
    } catch (e) {
      console.error('LiveKit streaming error', e);
      if (onToast) onToast(`LiveKit stream failed: ${e.message}`, 'error');
      setUseLiveKit(false);
    }
  };

  const handleWebRtcState = (state) => {
    if (state === 'connected') setWebRtcConnected(true);
    if (state === 'failed' || state === 'disconnected' || state === 'closed') {
      setWebRtcConnected(false);
      if (state === 'failed' && onToast) onToast('WebRTC connection failed — back to MP4', 'error');
      if (state === 'failed') handleStopWebRtc();
    }
  };

  const handleStartWebRtc = async () => {
    if (!generationResult?.request_id) return;
    try {
      setUseWebRtc(true);
      setWebRtcConnected(false);
      if (pcRef.current) pcRef.current.close();
      stopTracks(webRtcVideoRef.current?.srcObject);
      if (webRtcVideoRef.current) webRtcVideoRef.current.srcObject = null;

      const pc = await createWebRtcSession(
        generationResult.request_id,
        (event) => {
          // Audio and video can arrive as separate streams; merge the tracks
          // instead of letting the last ontrack replace the first.
          if (attachRemoteTrack(webRtcVideoRef.current, event)) {
            setWebRtcConnected(true);
          }
        },
        handleWebRtcState,
      );
      pcRef.current = pc;
      if (onToast) onToast('WebRTC peer stream established!', 'success');
    } catch (e) {
      console.error('WebRTC streaming error', e);
      if (onToast) onToast(`WebRTC stream failed: ${e.message}`, 'error');
      setUseWebRtc(false);
    }
  };

  const handleCopyLink = () => {
    if (!generationResult?.download_url) return;
    navigator.clipboard.writeText(generationResult.download_url);
    if (onToast) onToast('Video download URL copied to clipboard!', 'info');
  };

  // Cross-origin <a download> is ignored by browsers (navigates instead of
  // downloading), so fetch as blob and trigger a same-origin object URL.
  const handleDownload = async () => {
    if (!downloadUrl || !generationResult?.request_id || isDownloading) return;
    setIsDownloading(true);
    try {
      const res = await fetch(downloadUrl);
      if (!res.ok) throw new Error(`Download failed: ${res.statusText}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `float_talking_portrait_${generationResult.request_id}.mp4`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      if (onToast) onToast('Video download started!', 'success');
    } catch (e) {
      console.error('Download failed', e);
      // Fall back to plain navigation if the fetch path fails (e.g. CORS).
      window.open(downloadUrl, '_blank', 'noopener');
      if (onToast) onToast(`Direct download failed, opened link instead: ${e.message}`, 'info');
    } finally {
      setIsDownloading(false);
    }
  };

  // Single source of truth: the URLs the backend returned. The local API-base
  // reconstruction is only a fallback for stale results predating stream_url.
  const videoUrl = generationResult?.stream_url
    || (generationResult ? `${getApiUrl()}/stream/${generationResult.request_id}` : null)
    || '/samples/sam_altman_result.mp4';

  const downloadUrl = generationResult?.download_url
    || (generationResult ? `${getApiUrl()}/download/${generationResult.request_id}` : null);

  // Prefer real server progress when available; otherwise fall back to the
  // cosmetic stage cycler so the UI still moves during a single-pass run.
  const hasRealProgress = isGenerating && progress && typeof progress.percent === 'number';
  const progressPercent = hasRealProgress
    ? Math.max(0, Math.min(100, progress.percent))
    : ((currentStageIndex + 1) / GENERATION_STAGES.length) * 100;
  const progressLabel = hasRealProgress
    ? (progress.message
        || (progress.total_chunks > 0
          ? `${progress.completed_chunks}/${progress.total_chunks} chunks`
          : progress.stage))
    : `${GENERATION_STAGES[currentStageIndex].icon} ${GENERATION_STAGES[currentStageIndex].label}`;

  return (
    <div className="card-section glass-panel stage-panel">
      {/* Top CTA Banner */}
      <div className="stage-cta-bar">
        <div>
          <h2 className="stage-title">Studio Canvas & Output</h2>
          <p className="stage-desc">Real-time NVENC hardware-accelerated video rendering</p>
        </div>

        <button
          type="button"
          className="btn-primary btn-generate"
          onClick={onGenerate}
          disabled={isGenerating || !hasAudio}
        >
          {isGenerating ? (
            <>
              <div className="spinner-dot animate-spin" />
              <span>Generating Video...</span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span>Generate Portrait Video</span>
            </>
          )}
        </button>
      </div>

      {/* Transport selector */}
      <div className="transport-selector" role="group" aria-label="Stream transport">
        <span className="transport-selector-label">Transport</span>
        {TRANSPORTS.map((opt) => (
          <button
            key={opt.id}
            type="button"
            className={`transport-btn ${transport === opt.id ? 'is-active' : ''}`}
            onClick={() => setTransport?.(opt.id)}
            disabled={isGenerating}
            title={opt.title}
          >
            {opt.id !== 'mp4' && <span className="live-dot" />}
            {opt.label}
          </button>
        ))}
      </div>

      {/* Progress Overlay during Generation */}
      {isGenerating && (
        <div className="generation-progress-box">
          <div className="progress-header">
            <span className="stage-status-text">
              {progressLabel}
            </span>
            <span className="stage-step-count">
              {hasRealProgress ? `${Math.round(progressPercent)}%` : `${currentStageIndex + 1} / ${GENERATION_STAGES.length}`}
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill-bar"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      )}

      {/* Video Viewport */}
      <div className="video-viewport-wrapper">
        {isGenerating && liveRequestId && transport !== 'mp4' ? (
          <LivePreview
            requestId={liveRequestId}
            transport={transport}
            onStatus={(message) => onToast?.(message || 'Live stream ended', 'error')}
            onFallback={handleLiveFallback}
          />
        ) : useLiveKit ? (
          <div className="webrtc-player-box">
            <video
              ref={webRtcVideoRef}
              autoPlay
              playsInline
              controls
              className="stage-video"
            />
            <div className="webrtc-status-overlay">
              <span className="live-dot" />
              <span>{liveKitConnected ? 'LiveKit SFU Connected' : 'Connecting to LiveKit Room...'}</span>
              <button className="btn-secondary btn-xs" onClick={handleStopLiveKit}>
                Switch to Standard MP4
              </button>
            </div>
          </div>
        ) : useWebRtc ? (
          <div className="webrtc-player-box">
            <video
              ref={webRtcVideoRef}
              autoPlay
              playsInline
              controls
              className="stage-video"
            />
            <div className="webrtc-status-overlay">
              <span className="live-dot" />
              <span>{webRtcConnected ? 'WebRTC Live Connected' : 'Negotiating Peer Connection...'}</span>
              <button className="btn-secondary btn-xs" onClick={handleStopWebRtc}>
                Switch to Standard MP4
              </button>
            </div>
          </div>
        ) : (
          <div className="mp4-player-box">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              autoPlay
              loop
              playsInline
              className="stage-video"
            />
          </div>
        )}
      </div>

      {/* Action Toolbar below Video */}
      <div className="stage-toolbar">
        <div className="toolbar-left">
          {generationResult && (
            <div className="result-stat-chips">
              <span className="stat-chip">
                <Zap size={13} className="text-cyan" />
                <span>{generationResult.performance?.fps || 0} FPS</span>
              </span>
              <span className="stat-chip">
                <Gauge size={13} className="text-indigo" />
                <span>{generationResult.performance?.inference_time || 0}s Inf</span>
              </span>
              <span className="stat-chip">
                <Film size={13} />
                <span>{generationResult.file_size_mb || 0} MB</span>
              </span>
            </div>
          )}
        </div>

        <div className="toolbar-right">
          {generationResult && (
            <>
              {/* LiveKit Stream Toggle */}
              {!useLiveKit ? (
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  onClick={handleStartLiveKit}
                  title="Stream video via LiveKit SFU (WebRTC)"
                >
                  <Radio size={15} />
                  <span>LiveKit Stream</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  onClick={handleStopLiveKit}
                >
                  <span>Close LiveKit</span>
                </button>
              )}

              {/* WebRTC Peer Stream Toggle */}
              {!useWebRtc ? (
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  onClick={handleStartWebRtc}
                  title="Stream video via WebRTC peer connection"
                >
                  <Radio size={15} />
                  <span>WebRTC Stream</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  onClick={handleStopWebRtc}
                >
                  <span>Close WebRTC</span>
                </button>
              )}

              {/* Copy URL */}
              <button
                type="button"
                className="btn-icon"
                onClick={handleCopyLink}
                title="Copy Video Stream Link"
              >
                <Share2 size={16} />
              </button>

              {/* Download MP4 */}
              <button
                type="button"
                onClick={handleDownload}
                disabled={isDownloading}
                className="btn-primary btn-sm btn-download"
              >
                <Download size={15} />
                <span>{isDownloading ? 'Downloading...' : 'Download MP4'}</span>
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
