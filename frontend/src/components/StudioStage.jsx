import React, { useCallback, useRef, useState, useEffect } from 'react';
import {
  Download,
  Film,
  Gauge,
  Image as ImageIcon,
  Link2,
  Play,
  Radio,
  Sparkles,
  Wand2,
  Zap,
} from 'lucide-react';
import { getApiUrl, setIceServers } from '../services/api';
import { fetchLiveKitStatus } from '../services/livekit';
import LivePreview from './LivePreview';

const GENERATION_STAGES = [
  { label: 'Preprocessing audio & portrait', icon: '🎧' },
  { label: 'Encoding motion latent', icon: '⚡' },
  { label: 'Solving the flow ODE', icon: '🧠' },
  { label: 'Encoding video with NVENC', icon: '🎬' },
];

const TRANSPORTS = [
  { id: 'auto', label: 'Auto (recommended)' },
  { id: 'livekit', label: 'LiveKit (live preview)' },
  { id: 'webrtc', label: 'WebRTC (live preview)' },
  { id: 'ws', label: 'WebSocket (live frames)' },
  { id: 'mp4', label: 'File only (MP4)' },
];

export default function StudioStage({
  onGenerate,
  isGenerating,
  progress,
  generationResult,
  hasAudio,
  onToast,
  transport = 'auto',
  setTransport,
  liveRequestId,
  imagePreview,
}) {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [liveKitAvailable, setLiveKitAvailable] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [endedLiveRequestId, setEndedLiveRequestId] = useState(null);

  const onToastRef = useRef(onToast);
  useEffect(() => {
    onToastRef.current = onToast;
  }, [onToast]);

  // Check LiveKit availability on mount and whenever the active API URL changes.
  useEffect(() => {
    const checkLiveKit = () => {
      fetchLiveKitStatus()
        .then((status) => {
          setLiveKitAvailable(!!status.configured);
          // Feed the deployment's ICE list to the raw WebRTC transport.
          setIceServers(status.ice_servers?.length ? status.ice_servers : status.stun_urls);
        })
        .catch(() => setLiveKitAvailable(false));
    };
    checkLiveKit();
    window.addEventListener('float:api_url_changed', checkLiveKit);
    return () => window.removeEventListener('float:api_url_changed', checkLiveKit);
  }, []);

  // LiveKit is the primary SFU. When it is unconfigured or unreachable, switch
  // to raw WebRTC (signalled by this API host), not straight to the WebSocket.
  useEffect(() => {
    if (liveKitAvailable === false && transport === 'livekit') {
      setTransport?.('webrtc');
      onToastRef.current?.('LiveKit is unavailable — switching to WebRTC.', 'error');
    }
  }, [liveKitAvailable, transport, setTransport]);

  // Picking the MP4 transport is an explicit "leave the live view" action.
  const handleTransportChange = (next) => {
    if (next === 'mp4') setEndedLiveRequestId(liveRequestId ?? null);
    setTransport?.(next);
  };

  const handleLiveFallback = useCallback(
    (reason) => {
      const next = transport === 'livekit' ? 'webrtc' : 'ws';
      setTransport?.(next);
      const detail = reason ? ` (${reason})` : '';
      onToastRef.current?.(
        next === 'webrtc'
          ? `LiveKit failed${detail} — switched to WebRTC.`
          : `WebRTC failed${detail} — switched to low-latency frames.`,
        'error'
      );
    },
    [transport, setTransport]
  );

  // Cycle progress stages smoothly when the server has no real progress yet.
  useEffect(() => {
    let interval;
    if (isGenerating) {
      setCurrentStageIndex(0);
      interval = setInterval(() => {
        setCurrentStageIndex((prev) =>
          prev < GENERATION_STAGES.length - 1 ? prev + 1 : prev
        );
      }, 1600);
    } else {
      setCurrentStageIndex(0);
    }
    return () => clearInterval(interval);
  }, [isGenerating]);

  const handleCopyLink = () => {
    if (!generationResult?.download_url) return;
    navigator.clipboard.writeText(generationResult.download_url);
    onToast?.('Download link copied to clipboard.', 'info');
  };

  // Cross-origin <a download> is ignored by browsers, so fetch it as a blob.
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
      onToast?.('Download started.', 'success');
    } catch (e) {
      console.error('Download failed', e);
      window.open(downloadUrl, '_blank', 'noopener');
      onToast?.('Direct download failed — opened the file in a new tab.', 'info');
    } finally {
      setIsDownloading(false);
    }
  };

  const videoUrl = generationResult?.stream_url
    || (generationResult ? `${getApiUrl()}/stream/${generationResult.request_id}` : null);

  const downloadUrl = generationResult?.download_url
    || (generationResult ? `${getApiUrl()}/download/${generationResult.request_id}` : null);

  // Keep the live preview mounted after the POST returns: a real-time transport
  // is still playing frames buffered during the fast render.
  const livePreviewRequestId = (() => {
    if (!liveRequestId) return null;
    if (isGenerating) return liveRequestId;
    if (endedLiveRequestId === liveRequestId) return null;
    if (generationResult?.request_id !== liveRequestId) return null;
    return liveRequestId;
  })();

  const activeTransport =
    transport === 'auto'
      ? (liveKitAvailable === false ? 'webrtc' : 'livekit')
      : transport;

  const hasRealProgress = isGenerating && progress && typeof progress.percent === 'number';
  const progressPercent = hasRealProgress
    ? Math.max(0, Math.min(100, progress.percent))
    : ((currentStageIndex + 1) / GENERATION_STAGES.length) * 100;
  const progressLabel = hasRealProgress
    ? (progress.message
      || (progress.total_chunks > 0
        ? `${progress.completed_chunks}/${progress.total_chunks} chunks`
        : progress.stage))
    : GENERATION_STAGES[currentStageIndex].label;

  const showLivePreview = livePreviewRequestId && activeTransport !== 'mp4';

  return (
    <section className="glass-panel stage-panel">
      <div className="stage-head">
        <div className="stage-heading">
          <span className="stage-eyebrow">
            <Wand2 size={13} /> Output
          </span>
          <h2 className="stage-title">Studio canvas</h2>
          <p className="stage-desc">
            {isGenerating
              ? 'Frames appear here as soon as they are decoded.'
              : generationResult
                ? 'Your talking portrait is ready. Download it or generate another take.'
                : 'Your talking portrait will appear here.'}
          </p>
        </div>

        <button
          type="button"
          className="btn-primary btn-generate"
          onClick={onGenerate}
          disabled={isGenerating || !hasAudio}
          title={!hasAudio ? 'Add voice audio first' : 'Generate the talking portrait'}
        >
          {isGenerating ? (
            <>
              <span className="spinner-dot animate-spin" />
              <span>Generating…</span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span>{generationResult ? 'Generate another' : 'Generate video'}</span>
            </>
          )}
        </button>
      </div>

      {!hasAudio && !generationResult && (
        <div className="stage-hint">
          <span className="stage-hint-dot" />
          Record or upload a voice clip to unlock generation.
        </div>
      )}

      <div className={`stage-viewport ${isGenerating ? 'is-generating' : ''}`}>
        {showLivePreview ? (
          <LivePreview
            requestId={livePreviewRequestId}
            transport={activeTransport}
            onStatus={(message) => onToast?.(message || 'Live stream ended', 'error')}
            onFallback={handleLiveFallback}
            onEnded={(id) => setEndedLiveRequestId(id ?? livePreviewRequestId)}
          />
        ) : videoUrl ? (
          <video
            key={videoUrl}
            src={videoUrl}
            poster={imagePreview || undefined}
            controls
            autoPlay
            loop
            playsInline
            className="stage-video"
            onError={() => onToast?.('The rendered video could not be loaded. Try downloading the MP4.', 'error')}
          />
        ) : (
          <div className="stage-empty">
            <div
              className={`stage-empty-portrait ${imagePreview ? 'has-image' : ''}`}
              style={imagePreview ? { backgroundImage: `url(${imagePreview})` } : undefined}
            >
              {!imagePreview && <ImageIcon size={30} />}
              <span className="stage-empty-ring" />
            </div>
            <div className="stage-empty-copy">
              <span className="stage-empty-badge">
                <Play size={11} /> Ready
              </span>
              <h3>Bring a portrait to life</h3>
              <p>
                Add a voice clip, then press <strong>Generate video</strong>. With a live
                transport selected, frames start playing before the MP4 is finished.
              </p>
            </div>
          </div>
        )}

        {isGenerating && (
          <div className="stage-progress" role="status" aria-live="polite">
            <div className="stage-progress-top">
              <span className="stage-progress-label">
                <span className="stage-progress-pulse" />
                {progressLabel}
              </span>
              <span className="stage-progress-value">
                {hasRealProgress ? `${Math.round(progressPercent)}%` : `${currentStageIndex + 1} / ${GENERATION_STAGES.length}`}
              </span>
            </div>
            <div className="progress-track">
              <div className="progress-fill-bar" style={{ width: `${progressPercent}%` }} />
            </div>
          </div>
        )}
      </div>

      <div className="stage-toolbar">
        <div className="transport-control">
          <label htmlFor="transport-select" className="transport-label">
            <Radio size={13} />
            Preview
          </label>
          <select
            id="transport-select"
            className="transport-select"
            value={transport}
            onChange={(e) => handleTransportChange(e.target.value)}
            disabled={isGenerating}
            title="How the render is previewed"
          >
            {TRANSPORTS.map((opt) => (
              <option key={opt.id} value={opt.id}>{opt.label}</option>
            ))}
          </select>
        </div>

        {generationResult && (
          <div className="result-bar">
            <div className="result-stat-chips">
              <span className="stat-chip" title="Encoding speed">
                <Zap size={13} className="text-cyan" />
                {generationResult.performance?.fps || 0} FPS
              </span>
              <span className="stat-chip" title="Model inference time">
                <Gauge size={13} className="text-indigo" />
                {generationResult.performance?.inference_time || 0}s
              </span>
              <span className="stat-chip" title="File size">
                <Film size={13} />
                {generationResult.file_size_mb || 0} MB
              </span>
            </div>

            <button
              type="button"
              className="btn-icon"
              onClick={handleCopyLink}
              title="Copy download link"
              aria-label="Copy download link"
            >
              <Link2 size={16} />
            </button>
            <button
              type="button"
              onClick={handleDownload}
              disabled={isDownloading}
              className="btn-primary btn-sm"
            >
              <Download size={15} />
              <span>{isDownloading ? 'Downloading…' : 'Download MP4'}</span>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
