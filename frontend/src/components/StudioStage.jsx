import React, { useState, useRef, useEffect } from 'react';
import { 
  Play, 
  Sparkles, 
  Download, 
  Share2, 
  Radio, 
  Maximize2, 
  RotateCcw,
  CheckCircle2,
  Film,
  Zap,
  Gauge
} from 'lucide-react';
import { createWebRtcSession, getApiUrl } from '../services/api';

const GENERATION_STAGES = [
  { label: 'Preprocessing Audio & Image', icon: '🎧' },
  { label: 'Encoding Motion Latent Flow', icon: '⚡' },
  { label: 'Solving Transformer ODE Trajectory', icon: '🧠' },
  { label: 'NVENC Hardware Frame Compression', icon: '🎬' },
];

export default function StudioStage({
  onGenerate,
  isGenerating,
  generationResult,
  hasAudio,
  hasImage,
  onToast,
}) {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [useWebRtc, setUseWebRtc] = useState(false);
  const [webRtcConnected, setWebRtcConnected] = useState(false);

  const videoRef = useRef(null);
  const webRtcVideoRef = useRef(null);
  const pcRef = useRef(null);

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

  // Clean up WebRTC peer connection
  useEffect(() => {
    return () => {
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
    };
  }, []);

  const handleStartWebRtc = async () => {
    if (!generationResult?.request_id) return;
    try {
      setUseWebRtc(true);
      if (pcRef.current) pcRef.current.close();

      const pc = await createWebRtcSession(generationResult.request_id, (event) => {
        if (webRtcVideoRef.current && event.streams && event.streams[0]) {
          webRtcVideoRef.current.srcObject = event.streams[0];
          setWebRtcConnected(true);
        }
      });
      pcRef.current = pc;
      if (onToast) onToast('WebRTC peer stream established!', 'success');
    } catch (e) {
      console.error('WebRTC streaming error', e);
      if (onToast) onToast(`WebRTC stream failed: ${e.message}`, 'error');
      setUseWebRtc(false);
    }
  };

  const handleStopWebRtc = () => {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    setUseWebRtc(false);
    setWebRtcConnected(false);
  };

  const handleCopyLink = () => {
    if (!generationResult?.download_url) return;
    navigator.clipboard.writeText(generationResult.download_url);
    if (onToast) onToast('Video download URL copied to clipboard!', 'info');
  };

  const videoUrl = generationResult 
    ? `${getApiUrl()}/stream/${generationResult.request_id}`
    : '/samples/sam_altman_result.mp4';

  const downloadUrl = generationResult
    ? `${getApiUrl()}/download/${generationResult.request_id}`
    : null;

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

      {/* Progress Overlay during Generation */}
      {isGenerating && (
        <div className="generation-progress-box">
          <div className="progress-header">
            <span className="stage-status-text">
              {GENERATION_STAGES[currentStageIndex].icon} {GENERATION_STAGES[currentStageIndex].label}
            </span>
            <span className="stage-step-count">{currentStageIndex + 1} / {GENERATION_STAGES.length}</span>
          </div>
          <div className="progress-track">
            <div 
              className="progress-fill-bar"
              style={{ width: `${((currentStageIndex + 1) / GENERATION_STAGES.length) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Video Viewport */}
      <div className="video-viewport-wrapper">
        {useWebRtc ? (
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
              {/* WebRTC Live Stream Toggle */}
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
              <a 
                href={downloadUrl} 
                download={`float_talking_portrait_${generationResult.request_id}.mp4`}
                className="btn-primary btn-sm btn-download"
              >
                <Download size={15} />
                <span>Download MP4</span>
              </a>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
