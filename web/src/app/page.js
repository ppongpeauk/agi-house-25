'use client';

import dynamic from 'next/dynamic';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

const DynamicMap = dynamic(() => import('../components/Map'), {
  ssr: false,
  loading: () => (
    <div className="map-loading">
      <div className="pulse"></div>
    </div>
  ),
});

export default function Home() {
  return (
    <div className="disease-intel-platform">
      <nav className="top-nav">
        <div className="nav-left">
          <h1>DISEASE INTELLIGENCE</h1>
          <div className="status-indicator">LIVE</div>
        </div>
        <div className="nav-right">
          <div className="time-display">
            {new Date().toLocaleTimeString()}
          </div>
        </div>
      </nav>

      <main className="main-content">
        <div className="map-container">
          <DynamicMap />
          <div className="map-overlay">
            <div className="data-card temperature">
              <span className="label">TEMPERATURE</span>
              <span className="value" data-unit="°C">27</span>
              <div className="gauge"></div>
            </div>
            <div className="data-card humidity">
              <span className="label">HUMIDITY</span>
              <span className="value" data-unit="%">65</span>
              <div className="gauge"></div>
            </div>
            <div className="data-card wind">
              <span className="label">WIND SPEED</span>
              <span className="value" data-unit="km/h">12</span>
              <div className="gauge"></div>
            </div>
          </div>
        </div>

        <aside className="risk-panel">
          <div className="risk-header">
            <h2>RISK ANALYSIS</h2>
            <div className="confidence-meter">
              <span className="label">CONFIDENCE</span>
              <div className="meter">
                <div className="fill" style={{ width: '75%' }}></div>
              </div>
              <span className="value">75%</span>
            </div>
          </div>

          <div className="disease-status">
            <div className="status-card cholera">
              <span className="disease-name">CHOLERA</span>
              <span className="risk-level high">HIGH RISK</span>
            </div>
          </div>

          <div className="predictive-insights">
            <h3>PREDICTIVE INSIGHTS</h3>
            <div className="insight-card">
              <span className="trend">↗</span>
              <p>Risk increasing in southern regions due to rising temperatures</p>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
