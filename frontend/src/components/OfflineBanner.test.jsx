import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

vi.mock('../services/telemetry', () => ({
  useBackendStatus: vi.fn(),
  refreshBackendStatus: vi.fn(),
}));

import { useBackendStatus, refreshBackendStatus } from '../services/telemetry';
import OfflineBanner from './OfflineBanner';

beforeEach(() => {
  vi.clearAllMocks();
  refreshBackendStatus.mockResolvedValue(undefined);
});

describe('OfflineBanner', () => {
  it('stays hidden while the backend is healthy', () => {
    useBackendStatus.mockReturnValue({ online: true, offline: false, checking: false });
    const { container } = render(<OfflineBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it('does not flash during the first one or two failed polls', () => {
    useBackendStatus.mockReturnValue({ online: false, offline: false, checking: false });
    const { container } = render(<OfflineBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows a retry action after repeated failures', () => {
    useBackendStatus.mockReturnValue({ online: false, offline: true, checking: false });
    render(<OfflineBanner />);

    expect(screen.getByRole('alert')).toHaveTextContent(/backend unreachable/i);
    screen.getByRole('button', { name: /retry/i }).click();
    expect(refreshBackendStatus).toHaveBeenCalledTimes(1);
  });
});
