import React from 'react';
import { Sparkles } from 'lucide-react';

/**
 * The Studio's primary action.
 *
 * It lives at the bottom of the composer's last card, exactly where the Live
 * tab keeps `Go Live`: the left column owns every input *and* the button that
 * starts the render, so the right column stays a pure preview.
 */
export default function GenerateAction({
  onGenerate,
  isGenerating = false,
  hasAudio = false,
  hasResult = false,
}) {
  return (
    <div className="generate-action">
      <button
        type="button"
        className={`btn-primary btn-generate ${isGenerating ? 'is-busy' : ''}`}
        onClick={onGenerate}
        disabled={isGenerating || !hasAudio}
        aria-describedby={!hasAudio ? 'stage-audio-hint' : undefined}
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
            <span>{hasResult ? 'Generate another' : 'Generate video'}</span>
          </>
        )}
      </button>

      {!hasAudio && (
        <div className="stage-hint" id="stage-audio-hint">
          <span className="stage-hint-dot" aria-hidden="true" />
          {hasResult
            ? 'Add a voice clip to generate another take.'
            : 'Record or upload a voice clip to unlock generation.'}
        </div>
      )}
    </div>
  );
}
