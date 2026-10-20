import React, { useEffect } from 'react';
import { X, ShieldAlert, Clock, Video, Download } from 'lucide-react';
import { getEvidenceUrl } from '../api/client';
import { ActivityBadge } from './StatusBadge';

export const EvidenceModal = ({ event, onClose }) => {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!event) return null;

  const imageUrl = getEvidenceUrl(event.evidence_snapshot);
  const confidencePercent = (event.confidence * 100).toFixed(1);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-3xl bg-[#0F172A] border border-slate-700/80 rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0B0F17]/60">
          <div className="flex items-center gap-3">
            <ActivityBadge activity={event.activity_label} />
            <h3 id="modal-title" className="text-base font-semibold text-slate-100">
              Evidence Snapshot
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-lg transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Image Preview */}
        <div className="relative bg-black flex items-center justify-center min-h-[300px] max-h-[550px] overflow-hidden">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={`Evidence for ${event.activity_label}`}
              className="w-full h-auto max-h-[550px] object-contain"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect fill="%231E293B" width="400" height="300"/><text fill="%2394A3B8" font-family="sans-serif" font-size="14" x="50%" y="50%" text-anchor="middle">Evidence Snapshot Unavailable</text></svg>';
              }}
            />
          ) : (
            <div className="text-slate-500 flex flex-col items-center py-16">
              <ShieldAlert className="w-12 h-12 mb-2 text-slate-600" />
              <span>No evidence snapshot available</span>
            </div>
          )}
        </div>

        {/* Metadata Footer */}
        <div className="p-6 bg-[#0F172A] grid grid-cols-1 sm:grid-cols-3 gap-4 border-t border-slate-800">
          <div className="bg-[#1E293B]/60 p-3 rounded-xl border border-slate-700/40">
            <div className="text-xs text-slate-400 mb-1 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              Occurrence Time
            </div>
            <div className="text-sm font-semibold text-slate-100 font-mono">
              {event.start_time_formatted} &rarr; {event.end_time_formatted}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              ({event.start_timestamp}s - {event.end_timestamp}s)
            </div>
          </div>

          <div className="bg-[#1E293B]/60 p-3 rounded-xl border border-slate-700/40">
            <div className="text-xs text-slate-400 mb-1 flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
              Detection Confidence
            </div>
            <div className="text-sm font-semibold text-slate-100">
              {confidencePercent}%
            </div>
            <div className="w-full bg-slate-700/50 rounded-full h-1.5 mt-2 overflow-hidden">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${Math.min(100, Math.max(0, event.confidence * 100))}%` }}
              />
            </div>
          </div>

          <div className="bg-[#1E293B]/60 p-3 rounded-xl border border-slate-700/40">
            <div className="text-xs text-slate-400 mb-1 flex items-center gap-1.5">
              <Video className="w-3.5 h-3.5 text-indigo-400" />
              Source Video ID
            </div>
            <div className="text-xs font-mono text-slate-300 truncate" title={event.video_id}>
              {event.video_id}
            </div>
            {imageUrl && (
              <a
                href={imageUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex items-center gap-1 text-[11px] text-blue-400 hover:text-blue-300 transition-colors font-medium"
              >
                <Download className="w-3 h-3" />
                Open original image
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EvidenceModal;
