import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { SAMPLE_PORTRAITS } from '../services/samplePortraits';
import PortraitCanvas from './PortraitCanvas';

function renderCanvas(props = {}) {
  const setters = {
    setImagePreview: vi.fn(),
    setImageBase64: vi.fn(),
    setImageFormat: vi.fn(),
    setNoCrop: vi.fn(),
  };
  const utils = render(
    <PortraitCanvas
      imagePreview={null}
      noCrop={false}
      onToast={() => {}}
      {...setters}
      {...props}
    />,
  );
  return { ...utils, ...setters };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('PortraitCanvas samples', () => {
  it('renders one button per discovered portrait', () => {
    renderCanvas();

    for (const sample of SAMPLE_PORTRAITS) {
      expect(screen.getByRole('button', { name: new RegExp(sample.name, 'i') })).toBeInTheDocument();
    }
  });

  it('loads a sample as soon as it is clicked', async () => {
    const png = new Blob([new Uint8Array([1, 2, 3])], { type: 'image/jpeg' });
    global.fetch = vi.fn(async () => ({ ok: true, status: 200, blob: async () => png }));

    const { setImageBase64, setImagePreview, setImageFormat } = renderCanvas();
    const [first] = SAMPLE_PORTRAITS;

    fireEvent.click(screen.getByRole('button', { name: new RegExp(first.name, 'i') }));

    await waitFor(() => expect(setImageBase64).toHaveBeenCalled());
    expect(global.fetch).toHaveBeenCalledWith(first.url, expect.anything());
    expect(setImagePreview).toHaveBeenCalledWith(first.url);
    expect(setImageFormat).toHaveBeenCalledWith('jpg');
  });

  it('marks the active sample and reports failures instead of a broken image', async () => {
    global.fetch = vi.fn(async () => ({ ok: false, status: 404, blob: async () => new Blob([]) }));
    const onToast = vi.fn();
    const { setImageBase64 } = renderCanvas({
      imagePreview: SAMPLE_PORTRAITS[0].url,
      onToast,
    });

    const [first, second] = SAMPLE_PORTRAITS;
    expect(screen.getByRole('button', { name: new RegExp(first.name, 'i') }))
      .toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(screen.getByRole('button', { name: new RegExp(second.name, 'i') }));

    await waitFor(() => expect(onToast).toHaveBeenCalled());
    expect(setImageBase64).not.toHaveBeenCalled();
  });
});
