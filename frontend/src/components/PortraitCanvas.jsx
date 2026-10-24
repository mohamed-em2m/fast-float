import React, { useRef, useEffect } from 'react';
import { Upload, Image as ImageIcon, X, Crop } from 'lucide-react';
import { blobToBase64 } from '../services/api';

const SAMPLE_PORTRAITS = [
  { id: 'sam', name: 'Sam Altman', url: '/samples/portrait1.jpg' },
  { id: 'avatar', name: 'Studio Avatar', url: '/samples/portrait2.jpg' },
];

export default function PortraitCanvas({
  imagePreview,
  setImagePreview,
  setImageBase64,
  noCrop,
  setNoCrop,
  step = 1,
}) {
  const fileInputRef = useRef(null);
  // Tracks blob: preview URLs we created so we can revoke them (static
  // /samples/* URLs must NOT be revoked).
  const blobUrlRef = useRef(null);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  const setPreview = (url, { isBlob = false } = {}) => {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    if (isBlob) blobUrlRef.current = url;
    setImagePreview(url);
  };

  const handleFileUpload = async (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    setPreview(URL.createObjectURL(file), { isBlob: true });

    try {
      const base64 = await blobToBase64(file);
      setImageBase64(base64);
    } catch (e) {
      console.error('Failed to convert image to base64', e);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = async (sample) => {
    setPreview(sample.url);
    try {
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error(`Sample fetch failed: ${response.status}`);
      const blob = await response.blob();
      const base64 = await blobToBase64(blob);
      setImageBase64(base64);
    } catch (e) {
      console.error('Failed to load sample image', e);
    }
  };

  const handleClearImage = () => {
    setPreview(null);
    setImageBase64(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="card-section glass-panel">
      <div className="section-header">
        <div className="section-title-wrap">
          <span className="step-index">{step}</span>
          <ImageIcon size={18} className="section-icon text-indigo" />
          <h2 className="section-title">Portrait</h2>
        </div>
        {imagePreview && (
          <button className="btn-icon-subtle" onClick={handleClearImage} title="Clear image" aria-label="Clear portrait">
            <X size={16} />
          </button>
        )}
      </div>

      {/* Drag & Drop Canvas */}
      <div
        className={`portrait-dropzone ${imagePreview ? 'has-image' : ''}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => !imagePreview && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          style={{ display: 'none' }}
        />

        {imagePreview ? (
          <div className="portrait-preview-container">
            <img src={imagePreview} alt="Portrait Reference" className="portrait-preview-img" />
            <div className="portrait-overlay">
              <div className="portrait-focus-ring" />
              <button
                type="button"
                className="btn-change-image"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Change Photo
              </button>
            </div>
          </div>
        ) : (
          <div className="portrait-placeholder">
            <div className="upload-icon-circle">
              <Upload size={24} />
            </div>
            <p className="upload-hint-main">Drop portrait image here, or <span>browse</span></p>
            <p className="upload-hint-sub">PNG, JPG, WebP (Frontal face works best)</p>
          </div>
        )}
      </div>

      {/* Face Crop Toggle */}
      <div className="crop-toggle-row">
        <label className="crop-toggle-label">
          <input
            type="checkbox"
            checked={!noCrop}
            onChange={(e) => setNoCrop(!e.target.checked)}
            className="crop-checkbox"
          />
          <Crop size={15} />
          <span>Auto Face Crop & Centering</span>
        </label>
        <span className="crop-info-hint">Recommended for non-aligned photos</span>
      </div>

      {/* Sample Presets */}
      <div className="samples-row">
        <span className="samples-label">Quick Samples:</span>
        <div className="samples-pill-list">
          {SAMPLE_PORTRAITS.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className={`sample-pill ${imagePreview === sample.url ? 'active' : ''}`}
              onClick={() => handleSelectSample(sample)}
            >
              <img src={sample.url} alt={sample.name} className="sample-thumbnail" />
              <span>{sample.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
