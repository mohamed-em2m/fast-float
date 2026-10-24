import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Gauge,
  Shuffle,
  SlidersHorizontal,
  Sparkles,
  Zap,
} from 'lucide-react';

const EMOTIONS = [
  { id: 'S2E', label: 'Auto', icon: '✨', hint: 'Read the mood from the voice' },
  { id: 'neutral', label: 'Neutral', icon: '😐' },
  { id: 'happy', label: 'Happy', icon: '😊' },
  { id: 'sad', label: 'Sad', icon: '😢' },
  { id: 'angry', label: 'Angry', icon: '😠' },
  { id: 'surprise', label: 'Surprise', icon: '😲' },
  { id: 'disgust', label: 'Disgust', icon: '🤢' },
];

const QUALITY_PRESETS = [
  { id: 'draft', label: 'Fast', nfe: 6, hint: 'Fastest preview' },
  { id: 'balanced', label: 'Balanced', nfe: 10, hint: 'Recommended' },
  { id: 'cinematic', label: 'Cinematic', nfe: 16, hint: 'Best detail' },
];

export default function ParameterControls({ params, setParams, step = 3 }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateParam = (key, val) => {
    setParams((prev) => ({ ...prev, [key]: val }));
  };

  const handleRandomizeSeed = () => {
    updateParam('seed', Math.floor(Math.random() * 100000));
  };

  const activePreset = QUALITY_PRESETS.reduce((closest, preset) => {
    if (!closest) return preset;
    return Math.abs(preset.nfe - params.nfe) < Math.abs(closest.nfe - params.nfe)
      ? preset
      : closest;
  }, null);

  return (
    <div className="card-section glass-panel">
      <div className="section-header">
        <div className="section-title-wrap">
          <span className="step-index">{step}</span>
          <SlidersHorizontal size={18} className="section-icon text-indigo" />
          <h2 className="section-title">Style</h2>
        </div>
        <span className="muted-label">optional</span>
      </div>

      {/* Emotion Selector */}
      <div className="control-group">
        <label className="control-label">
          <span>Emotion</span>
          <span className="control-val-badge">
            {EMOTIONS.find((emotion) => emotion.id === params.emotion)?.label || params.emotion}
          </span>
        </label>
        <div className="emotion-grid">
          {EMOTIONS.map((emo) => (
            <button
              key={emo.id}
              type="button"
              className={`emotion-btn ${params.emotion === emo.id ? 'active' : ''}`}
              onClick={() => updateParam('emotion', emo.id)}
              title={emo.hint || `Force a ${emo.label.toLowerCase()} expression`}
              aria-pressed={params.emotion === emo.id}
            >
              <span className="emotion-icon">{emo.icon}</span>
              <span className="emotion-name">{emo.label}</span>
            </button>
          ))}
        </div>
        <p className="control-subtext">
          Auto follows the speaker. Pick a face to redirect the performance.
        </p>
      </div>

      {/* Quality Presets */}
      <div className="control-group">
        <div className="control-label-row">
          <label className="control-label">
            <span>Render quality</span>
            <span className="control-val-badge">{params.nfe} steps</span>
          </label>
        </div>
        <div className="preset-grid">
          {QUALITY_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={`preset-btn ${activePreset?.id === preset.id ? 'active' : ''}`}
              onClick={() => updateParam('nfe', preset.nfe)}
              aria-pressed={activePreset?.id === preset.id}
            >
              {preset.id === 'draft' ? <Zap size={14} /> : preset.id === 'cinematic' ? <Sparkles size={14} /> : <Gauge size={14} />}
              <span>{preset.label}</span>
              <small>{preset.hint}</small>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Toggle */}
      <div className="advanced-toggle-row">
        <button
          type="button"
          className="btn-text-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
          aria-expanded={showAdvanced}
        >
          <span>Fine-tune &amp; performance</span>
          {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Advanced Settings Drawer */}
      {showAdvanced && (
        <div className="advanced-drawer">
          {/* NFE Steps Slider */}
          <div className="control-group">
            <div className="control-label-row">
              <label className="control-label">
                <span>Sampling steps (NFE)</span>
                <span className="control-val-badge">{params.nfe}</span>
              </label>
              <span className="control-subtext">
                {params.nfe <= 8 ? '⚡ Fastest' : params.nfe <= 15 ? 'Balanced' : 'High fidelity'}
              </span>
            </div>
            <input
              type="range"
              min="5"
              max="25"
              step="1"
              value={params.nfe}
              onChange={(e) => updateParam('nfe', parseInt(e.target.value, 10))}
              aria-label="Sampling steps"
            />
            <div className="slider-ticks">
              <span>5 · fastest</span>
              <span>25 · max quality</span>
            </div>
          </div>

          {/* Seed Controls */}
          <div className="control-group">
            <label className="control-label">
              <span>Seed</span>
              <span className="control-subtext">Lock a look across takes</span>
            </label>
            <div className="seed-input-row">
              <input
                type="number"
                value={params.seed}
                onChange={(e) => updateParam('seed', parseInt(e.target.value, 10) || 0)}
                className="seed-input"
                aria-label="Generation seed"
              />
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={handleRandomizeSeed}
                title="Randomize seed"
              >
                <Shuffle size={14} />
                <span>Random</span>
              </button>
            </div>
          </div>

          {/* Chunking */}
          <div className="control-group">
            <div className="control-label-row">
              <label className="control-label">
                <span>Long speech chunking</span>
                <span className="control-val-badge">
                  {params.chunk_seconds > 0 ? `${params.chunk_seconds}s windows` : 'Off'}
                </span>
              </label>
              <button
                type="button"
                className={`switch ${params.chunk_seconds > 0 ? 'is-on' : ''}`}
                onClick={() => updateParam('chunk_seconds', params.chunk_seconds > 0 ? 0 : 5)}
                aria-pressed={params.chunk_seconds > 0}
                aria-label="Toggle long speech chunking"
              >
                <span />
              </button>
            </div>
            {params.chunk_seconds > 0 && (
              <>
                <input
                  type="range"
                  min="1"
                  max="30"
                  step="0.5"
                  value={params.chunk_seconds}
                  onChange={(e) => updateParam('chunk_seconds', parseFloat(e.target.value))}
                  aria-label="Chunk window length"
                />
                <div className="slider-ticks">
                  <span>1s</span>
                  <span>30s</span>
                </div>
                <div className="control-label-row" style={{ marginTop: '6px' }}>
                  <label className="control-label">
                    <span>Chunks per batch</span>
                    <span className="control-val-badge">{params.chunk_batch_size}×</span>
                  </label>
                </div>
                <input
                  type="range"
                  min="1"
                  max="8"
                  step="1"
                  value={params.chunk_batch_size}
                  onChange={(e) => updateParam('chunk_batch_size', parseInt(e.target.value, 10))}
                  aria-label="Chunks per batch"
                />
                <p className="control-subtext">
                  Faster on long clips; needs VRAM headroom.
                </p>
              </>
            )}
          </div>

          {/* Guidance Scales */}
          <div className="control-group">
            <label className="control-label">
              <span>Guidance</span>
              <span className="control-val-badge">A / R / E</span>
            </label>
            <div className="guidance-grid">
              <label>
                <span>Audio {params.a_cfg_scale.toFixed(1)}</span>
                <input
                  type="range"
                  min="1"
                  max="4"
                  step="0.1"
                  value={params.a_cfg_scale}
                  onChange={(e) => updateParam('a_cfg_scale', parseFloat(e.target.value))}
                />
              </label>
              <label>
                <span>Identity {params.r_cfg_scale.toFixed(1)}</span>
                <input
                  type="range"
                  min="0.5"
                  max="3"
                  step="0.1"
                  value={params.r_cfg_scale}
                  onChange={(e) => updateParam('r_cfg_scale', parseFloat(e.target.value))}
                />
              </label>
              <label>
                <span>Emotion {params.e_cfg_scale.toFixed(1)}</span>
                <input
                  type="range"
                  min="0.5"
                  max="3"
                  step="0.1"
                  value={params.e_cfg_scale}
                  onChange={(e) => updateParam('e_cfg_scale', parseFloat(e.target.value))}
                />
              </label>
            </div>
          </div>

          {/* Encoding Quality */}
          <div className="control-group">
            <label className="control-label">
              <span>Encoding</span>
              <span className="control-subtext">NVENC preset</span>
            </label>
            <div className="quality-radio-group">
              {['fast', 'balanced', 'high'].map((q) => (
                <button
                  key={q}
                  type="button"
                  className={`quality-btn ${params.quality === q ? 'active' : ''}`}
                  onClick={() => updateParam('quality', q)}
                  aria-pressed={params.quality === q}
                >
                  <span>{q.toUpperCase()}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Decode Batch Size */}
          <div className="control-group">
            <div className="control-label-row">
              <label className="control-label">
                <span>Decode batch</span>
                <span className="control-val-badge">
                  {params.decode_batch_size > 0 ? `${params.decode_batch_size} frames` : 'Auto'}
                </span>
              </label>
              <span className="control-subtext">
                {params.decode_batch_size > 0 ? '⚠ Manual VRAM' : 'VRAM auto-tuned'}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="64"
              step="1"
              value={params.decode_batch_size}
              onChange={(e) => updateParam('decode_batch_size', parseInt(e.target.value, 10))}
              aria-label="Decode batch size"
            />
            <div className="slider-ticks">
              <span>0 · auto</span>
              <span>64 · max</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
