'use client';

import dynamic from 'next/dynamic';
import { Inter } from 'next/font/google';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DiseaseRiskPanel } from '../components/disease-risk-panel';

const inter = Inter({ subsets: ['latin'] });

const DynamicMap = dynamic(() => import('../components/Map'), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <div className="w-12 h-12 bg-cyan-500 rounded-full animate-pulse" />
    </div>
  ),
});

// Create a client
const queryClient = new QueryClient();

export default function Home() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-black text-white flex flex-col">
        {/* Navigation */}
        <nav className="h-14 border-b border-white/10 bg-black/50 backdrop-blur-lg flex items-center px-6 sticky top-0 z-50">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-bold">
              <span className="text-cyan-500">DISEASE</span>{' '}
              <span className="text-white">INTELLIGENCE</span>
            </h1>
            <div className="px-2 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-500">
              LIVE
            </div>
          </div>
          <div className="ml-auto text-sm text-white/70">
            {new Date().toLocaleTimeString()}
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 flex">
          {/* Map Section */}
          <div className="relative flex-1 bg-zinc-900">
            <DynamicMap />

            {/* Overlay Cards */}
            <div className="absolute top-4 left-4 space-y-2 z-10">
              <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4 group hover:bg-black/90 transition-colors">
                <div className="flex items-center gap-2 text-xs text-white/60 mb-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                  TEMPERATURE
                </div>
                <div className="text-3xl font-bold text-cyan-500 flex items-baseline gap-1">
                  27<span className="text-sm text-white/40">°C</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden">
                  <div className="h-full w-3/4 bg-gradient-to-r from-cyan-500 to-pink-500 rounded-full" />
                </div>
              </div>

              <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4 group hover:bg-black/90 transition-colors">
                <div className="flex items-center gap-2 text-xs text-white/60 mb-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                  HUMIDITY
                </div>
                <div className="text-3xl font-bold text-cyan-500 flex items-baseline gap-1">
                  65<span className="text-sm text-white/40">%</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden">
                  <div className="h-full w-2/3 bg-gradient-to-r from-cyan-500 to-pink-500 rounded-full" />
                </div>
              </div>

              <div className="w-56 bg-black/80 backdrop-blur-lg border border-white/10 rounded-lg p-4 group hover:bg-black/90 transition-colors">
                <div className="flex items-center gap-2 text-xs text-white/60 mb-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                  WIND SPEED
                </div>
                <div className="text-3xl font-bold text-cyan-500 flex items-baseline gap-1">
                  12<span className="text-sm text-white/40">km/h</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full mt-3 overflow-hidden">
                  <div className="h-full w-1/2 bg-gradient-to-r from-cyan-500 to-pink-500 rounded-full" />
                </div>
              </div>
            </div>
          </div>

          {/* Risk Panel */}
          <aside className="w-[400px] bg-black/50 backdrop-blur-lg border-l border-white/10">
            <DiseaseRiskPanel />
          </aside>
        </main>
      </div>
    </QueryClientProvider>
  );
}
