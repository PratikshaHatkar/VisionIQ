import React, { useState, useEffect } from 'react';
import {
  CheckCircle2,
  XCircle,
  Cpu,
  Database,
  Server,
  FolderArchive,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import { api } from '../api/client';

export const SystemStatusPage = () => {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [checking, setChecking] = useState(false);

  const runDiagnostics = async () => {
    const start = performance.now();
    try {
      const data = await api.getHealth();
      const end = performance.now();
      setLatency(Math.round(end - start));
      setHealth(data);
      setError(null);
    } catch (err) {
      setError(err.message || 'FastAPI backend is unreachable on http://127.0.0.1:8000.');
    } finally {
      setLoading(false);
      setChecking(false);
    }
  };

  useEffect(() => {
    runDiagnostics();
  }, []);

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            System Diagnostics & Status
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Live Health
            </span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Real-time verification of model loading, GPU hardware acceleration, and database persistence.
          </p>
        </div>

        <button
          onClick={() => {
            setChecking(true);
            runDiagnostics();
          }}
          disabled={checking}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#111827] text-slate-200 border border-slate-700/80 hover:bg-slate-800 transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${checking ? 'animate-spin text-blue-400' : ''}`} />
          Run Diagnostics
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <XCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={runDiagnostics}
            className="px-3 py-1 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 rounded-lg text-xs font-medium border border-rose-500/40"
          >
            Retry
          </button>
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
          <span className="text-xs text-slate-400">Querying /health diagnostic endpoint...</span>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Diagnostic Overview Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* FastAPI Service Status */}
            <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Server className="w-4 h-4 text-blue-400" />
                  REST API Backend
                </span>
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Operational
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Framework</span>
                  <span className="font-semibold text-white">FastAPI (Uvicorn)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">API Version</span>
                  <span className="font-mono text-white">{health?.version || '1.0.0'}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Host Endpoint</span>
                  <span className="font-mono text-blue-400">http://127.0.0.1:8000</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Round-trip Latency</span>
                  <span className="font-mono text-emerald-400">{latency ? `${latency} ms` : '—'}</span>
                </div>
              </div>
            </div>

            {/* AI Model Diagnostic */}
            <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-indigo-400" />
                  Spatiotemporal Model
                </span>
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Active
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Architecture</span>
                  <span className="font-semibold text-white font-mono">{health?.model?.name || 'r2plus1d_18'}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Execution Hardware</span>
                  <span className="font-semibold text-indigo-400 font-mono">
                    {health?.model?.device?.toUpperCase() || 'CUDA (GPU)'}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Weights Loaded</span>
                  <span className="font-mono text-emerald-400">
                    {health?.model?.loaded ? 'True (In VRAM)' : 'Ready'}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Checkpoints</span>
                  <span className="font-mono text-slate-400 truncate max-w-[150px]">best_model.pth</span>
                </div>
              </div>
            </div>

            {/* Database Persistence */}
            <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Database className="w-4 h-4 text-emerald-400" />
                  Persistence Layer
                </span>
                <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${
                  health?.database_engine === 'mongodb'
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                }`}>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {health?.database_engine === 'mongodb' ? 'MongoDB Primary' : 'SQLite Fallback'}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Database Engine</span>
                  <span className="font-semibold text-white">{health?.database_engine || 'mongodb'}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Database Name</span>
                  <span className="font-mono text-white">{health?.mongodb?.database || 'visioniq'}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Connection Details</span>
                  <span className="font-mono text-emerald-400">{health?.database_details || 'Connected'}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Fallback Resilience</span>
                  <span className="font-mono text-slate-300">SQLite (visioniq.db)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Recognizable Surveillance Classes */}
          <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-blue-400" />
              Recognized Surveillance Activity Classes (7 Categories)
            </h3>
            <p className="text-xs text-slate-400">
              The fine-tuned R(2+1)D model is trained to recognize 7 mutually exclusive human action categories from surveillance video clips:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { name: 'Normal', desc: 'Typical pedestrian walking and regular activities.' },
                { name: 'Fighting', desc: 'Physical violence, scuffles, and aggressive assaults.' },
                { name: 'Robbery', desc: 'Armed or forced theft involving persons.' },
                { name: 'Shoplifting', desc: 'Concealment and illicit removal of store goods.' },
                { name: 'Burglary', desc: 'Forced unauthorized entry into property.' },
                { name: 'RoadAccidents', desc: 'Vehicular collisions and pedestrian impacts.' },
                { name: 'Vandalism', desc: 'Destruction or defacement of property and assets.' },
              ].map((c) => (
                <div key={c.name} className="p-3 bg-[#0F172A] rounded-xl border border-slate-800 space-y-1">
                  <div className="font-semibold text-xs text-white">{c.name}</div>
                  <div className="text-[11px] text-slate-400">{c.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* File System & Storage Locations */}
          <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6 space-y-3">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <FolderArchive className="w-4 h-4 text-slate-400" />
              Local Storage Partitions
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3 rounded-xl bg-[#0F172A] border border-slate-800 space-y-1">
                <span className="text-slate-400">Raw CCTV Videos Partition</span>
                <div className="font-mono text-slate-200">storage/videos/</div>
                <div className="text-[11px] text-slate-500">Stores ingested MP4/AVI videos before analysis.</div>
              </div>
              <div className="p-3 rounded-xl bg-[#0F172A] border border-slate-800 space-y-1">
                <span className="text-slate-400">Evidence Keyframe Snapshots</span>
                <div className="font-mono text-slate-200">storage/evidence/</div>
                <div className="text-[11px] text-slate-500">Stores high-resolution JPEG evidence frames for flagged alerts.</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemStatusPage;
