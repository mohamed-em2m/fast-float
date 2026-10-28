import React from 'react';
import { AlertTriangle, Copy, RotateCw } from 'lucide-react';
import { copyText } from '../services/clipboard';

function buildDiagnostics(error, componentStack) {
  return [
    'FLOAT Studio — render error',
    `Time: ${new Date().toISOString()}`,
    `URL: ${typeof window !== 'undefined' ? window.location.href : '(unknown)'}`,
    `User agent: ${typeof navigator !== 'undefined' ? navigator.userAgent : '(unknown)'}`,
    '',
    `Error: ${error?.name || 'Error'}: ${error?.message || String(error)}`,
    error?.stack ? `\nStack:\n${error.stack}` : '',
    componentStack ? `\nComponent stack:${componentStack}` : '',
  ].join('\n');
}

/**
 * Last line of defence: a render error in any panel must not blank the whole
 * Studio. The fallback keeps the message readable and offers a reload plus a
 * diagnostics copy button for bug reports.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null, componentStack: '', copied: false };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[studio] render error', error, info?.componentStack || '');
    this.setState({ componentStack: info?.componentStack || '' });
  }

  handleReload = () => {
    if (typeof window !== 'undefined') window.location.reload();
  };

  handleCopy = async () => {
    const ok = await copyText(buildDiagnostics(this.state.error, this.state.componentStack));
    this.setState({ copied: ok });
    if (ok) setTimeout(() => this.setState({ copied: false }), 3000);
  };

  render() {
    const { error, copied } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="crash-overlay" role="alert">
        <section className="crash-panel glass-panel">
          <div className="crash-icon">
            <AlertTriangle size={26} />
          </div>
          <h1 className="crash-title">The Studio hit an unexpected error</h1>
          <p className="crash-copy">
            Your inputs are still in this tab. Reload to recover; if it happens again,
            copy the diagnostics and include them in a bug report.
          </p>
          <pre className="crash-detail">{error.message || String(error)}</pre>
          <div className="crash-actions">
            <button type="button" className="btn-primary" onClick={this.handleReload}>
              <RotateCw size={16} />
              Reload Studio
            </button>
            <button type="button" className="btn-secondary" onClick={this.handleCopy}>
              <Copy size={15} />
              {copied ? 'Copied diagnostics' : 'Copy diagnostics'}
            </button>
          </div>
        </section>
      </div>
    );
  }
}
