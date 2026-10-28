import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';

function Boom() {
  throw new Error('kaboom');
}

describe('ErrorBoundary', () => {
  it('keeps the app shell alive and explains how to recover', () => {
    // React logs the caught render error; keep the test output readable.
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/unexpected error/i)).toBeInTheDocument();
    expect(screen.getByText('kaboom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reload studio/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /copy diagnostics/i })).toBeInTheDocument();

    consoleError.mockRestore();
  });

  it('renders children while healthy', () => {
    render(
      <ErrorBoundary>
        <p>Studio ready</p>
      </ErrorBoundary>
    );
    expect(screen.getByText('Studio ready')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
