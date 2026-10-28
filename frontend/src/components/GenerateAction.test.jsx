import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import GenerateAction from './GenerateAction';

function renderAction(props = {}) {
  return render(
    <GenerateAction
      onGenerate={() => {}}
      isGenerating={false}
      hasAudio
      hasResult={false}
      {...props}
    />
  );
}

describe('GenerateAction', () => {
  it('starts a render from the composer', () => {
    const onGenerate = vi.fn();
    renderAction({ onGenerate });

    screen.getByRole('button', { name: /generate video/i }).click();
    expect(onGenerate).toHaveBeenCalledTimes(1);
  });

  it('explains why generation is locked when no voice clip is available', () => {
    renderAction({ hasAudio: false });

    const button = screen.getByRole('button', { name: /generate video/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('aria-describedby', 'stage-audio-hint');
    expect(screen.getByText(/record or upload a voice clip/i)).toBeInTheDocument();
  });

  it('keeps the hint after a result when the audio is cleared', () => {
    renderAction({ hasAudio: false, hasResult: true });

    expect(screen.getByText(/add a voice clip to generate another take/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate another/i })).toBeDisabled();
  });

  it('swaps to a busy label while the render runs', () => {
    renderAction({ isGenerating: true, hasResult: true });

    const button = screen.getByRole('button', { name: /generating/i });
    expect(button).toBeDisabled();
  });
});
