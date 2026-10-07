import React, { useState } from 'react';
import { Sliders, Dna, Shuffle, Zap, ChevronDown, ChevronUp } from 'lucide-react';

const EMOTIONS = [
  { id: 'S2E', label: 'Speech-to-Emotion', icon: '🧠' },
  { id: 'neutral', label: 'Neutral', icon: '😐' },
  { id: 'happy', label: 'Happy', icon: '😊' },
  { id: 'sad', label: 'Sad', icon: '😢' },
  { id: 'angry', label: 'Angry', icon: '😠' },
  { id: 'surprise', label: 'Surprise', icon: '😲' },
  { id: 'disgust', label: 'Disgust', icon: '🤢' },
];

export default function ParameterControls({ params, setParams }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateParam = (key, val) => {
    setParams((prev) => ({ ...prev, [key]: val }));
  };

  const handleRandomizeSeed = () => {
    updateParam('seed', Math.floor(Math.random() * 100000));
  };

  return (
    <div className="card-section glass-panel">
      <div className="section-header">
        <div className="section-title-wrap">
          <Sliders size={18} className="section-icon text-indigo" />
          <h2 className="section-title">Generation Parameters</h2>
        </div>
      </div>

      {/* Emotion Selector */}
      <div className="control-group">
        <label className="control-label">
          <span>Emotion Mode</span>
          <span className="control-val-badge">{params.emotion}</span>
        </label>
        <div className="emotion-grid">
          {EMOTIONS.map((emo) => (
            <button
              key={emo.id}
              type="button"
              className={`emotion-btn ${params.emotion === emo.id ? 'active' : ''}`}
              onClick={() => updateParam('emotion', emo.id)}
            >
              <span className="emotion-icon">{emo.icon}</span>
              <span className="emotion-name">{emo.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* NFE Steps Slider */}
      <div className="control-group">
        <div className="control-label-row">
          <label className="control-label">
            <span>Sampling Steps (NFE)</span>
            <span className="control-val-badge">{params.nfe} steps</span>
          </label>
          <span className="control-subtext">
            {params.nfe <= 8 ? '⚡ Ultra-fast' : params.nfe <= 15 ? 'Balanced (Recommended)' : 'High Fidelity'}
          </span>
        </div>
        <input 
          type="range" 
          min="5" 
          max="25" 
          step="1"
          value={params.nfe} 
          onChange={(e) => updateParam('nfe', parseInt(e.target.value, 10))}
        />
        <div className="slider-ticks">
          <span>5 (Fastest)</span>
          <span>10 (Default)</span>
          <span>25 (Max Quality)</span>
        </div>
      </div>

      {/* Seed Controls */}
      <div className="control-group">
        <label className="control-label">
          <span>Generation Seed</span>
        </label>
        <div className="seed-input-row">
          <input 
            type="number" 
            value={params.seed} 
            onChange={(e) => updateParam('seed', parseInt(e.target.value, 10) || 0)} 
            className="seed-input"
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

      {/* Advanced Toggle */}
      <div className="advanced-toggle-row">
        <button 
          type="button" 
          className="btn-text-toggle" 
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <span>Advanced Guidance & Compression</span>
          {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Advanced Settings Drawer */}
      {showAdvanced && (
        <div className="advanced-drawer">
          {/* Audio CFG Scale */}
          <div className="control-group">
            <div className="control-label-row">
              <label className="control-label">
                <span>Audio CFG Scale</span>
                <span className="control-val-badge">{params.a_cfg_scale.toFixed(1)}</span>
              </label>
            </div>
            <input 
              type="range" 
              min="1.0" 
              max="4.0" 
              step="0.1"
              value={params.a_cfg_scale} 
              onChange={(e) => updateParam('a_cfg_scale', parseFloat(e.target.value))}
            />
          </div>

          {/* Reference CFG Scale */}
          <div className="control-group">
            <div className="control-label-row">
              <label className="control-label">
                <span>Reference Identity CFG</span>
                <span className="control-val-badge">{params.r_cfg_scale.toFixed(1)}</span>
              </label>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="3.0" 
              step="0.1"
              value={params.r_cfg_scale} 
              onChange={(e) => updateParam('r_cfg_scale', parseFloat(e.target.value))}
            />
          </div>

          {/* Emotion CFG Scale */}
          <div className="control-group">
            <div className="control-label-row">
              <label className="control-label">
                <span>Emotion CFG Scale</span>
                <span className="control-val-badge">{params.e_cfg_scale.toFixed(1)}</span>
              </label>
            </div>
            <input 
              type="range" 
              min="0.5" 
              max="3.0" 
              step="0.1"
              value={params.e_cfg_scale} 
              onChange={(e) => updateParam('e_cfg_scale', parseFloat(e.target.value))}
            />
          </div>

          {/* Encoding Quality */}
          <div className="control-group">
            <label className="control-label">
              <span>NVENC Quality Preset</span>
            </label>
            <div className="quality-radio-group">
              {['fast', 'balanced', 'high'].map((q) => (
                <button
                  key={q}
                  type="button"
                  className={`quality-btn ${params.quality === q ? 'active' : ''}`}
                  onClick={() => updateParam('quality', q)}
                >
                  <Zap size={13} />
                  <span>{q.toUpperCase()}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
