import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  ShieldAlert,
  Filter,
  Video,
  Eye,
  RefreshCw,
  X,
  LayoutGrid,
  List,
  AlertCircle,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { api, getEvidenceUrl } from '../api/client';
import { ActivityBadge } from '../components/StatusBadge';
import EvidenceModal from '../components/EvidenceModal';
import EmptyState from '../components/EmptyState';

const ACTIVITY_OPTIONS = [
  'ALL',
  'Fighting',
  'Robbery',
  'Burglary',
  'Shoplifting',
  'Vandalism',
  'RoadAccidents',
];

export const EventsPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  // Filters from URL or defaults
  const [activity, setActivity] = useState(searchParams.get('activity') || 'ALL');
  const [videoId, setVideoId] = useState(searchParams.get('videoId') || '');
  const [minConfidence, setMinConfidence] = useState(parseFloat(searchParams.get('minConfidence')) || 0);

  // Pagination
  const [page, setPage] = useState(0);
  const pageSize = 24;

  // View mode: 'grid' or 'table'
  const [viewMode, setViewMode] = useState('grid');

  // Data state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [eventsData, setEventsData] = useState({ total: 0, events: [] });
  const [selectedEvent, setSelectedEvent] = useState(null);

  const fetchEvents = useCallback(async () => {
    try {
      const data = await api.getEvents({
        activity: activity === 'ALL' ? undefined : activity,
        videoId: videoId.trim() || undefined,
        minConfidence: minConfidence > 0 ? minConfidence : undefined,
        limit: pageSize,
        offset: page * pageSize,
      });
      setEventsData(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch detected events.');
    } finally {
      setLoading(false);
    }
  }, [activity, videoId, minConfidence, page, pageSize]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  // Sync state to URL search params
  const handleApplyFilters = (newActivity, newVideoId, newMinConfidence) => {
    setPage(0);
    const params = {};
    if (newActivity && newActivity !== 'ALL') params.activity = newActivity;
    if (newVideoId && newVideoId.trim()) params.videoId = newVideoId.trim();
    if (newMinConfidence > 0) params.minConfidence = newMinConfidence.toString();
    setSearchParams(params);
  };

  const handleResetFilters = () => {
    setActivity('ALL');
    setVideoId('');
    setMinConfidence(0);
    setPage(0);
    setSearchParams({});
  };

  const totalPages = Math.ceil((eventsData.total || 0) / pageSize);

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            Detected Suspicious Events
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20">
              {eventsData.total} Total Alerts
            </span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Browse and inspect genuine action recognition detections across all CCTV recordings.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* View toggle */}
          <div className="flex items-center bg-[#111827] border border-slate-800 rounded-xl p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1.5 rounded-lg transition-colors ${
                viewMode === 'grid' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
              title="Grid View"
            >
              <LayoutGrid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`p-1.5 rounded-lg transition-colors ${
                viewMode === 'table' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
              title="Table View"
            >
              <List className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={() => {
              setLoading(true);
              fetchEvents();
            }}
            disabled={loading}
            className="p-2 bg-[#111827] hover:bg-slate-800 text-slate-300 rounded-xl border border-slate-800 transition-all disabled:opacity-50"
            title="Refresh Events"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-blue-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="bg-[#111827] border border-slate-800 rounded-2xl p-4 shadow-sm space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Activity Category Dropdown */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5 flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-blue-400" />
              Activity Category
            </label>
            <select
              value={activity}
              onChange={(e) => {
                setActivity(e.target.value);
                handleApplyFilters(e.target.value, videoId, minConfidence);
              }}
              className="w-full bg-[#0F172A] border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
            >
              {ACTIVITY_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt === 'ALL' ? 'All Activities' : opt}
                </option>
              ))}
            </select>
          </div>

          {/* Video ID search */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5 flex items-center gap-1.5">
              <Video className="w-3.5 h-3.5 text-indigo-400" />
              Video ID Filter
            </label>
            <div className="relative">
              <input
                type="text"
                placeholder="Search by Video UUID..."
                value={videoId}
                onChange={(e) => {
                  setVideoId(e.target.value);
                  handleApplyFilters(activity, e.target.value, minConfidence);
                }}
                className="w-full bg-[#0F172A] border border-slate-700/80 rounded-xl pl-3 pr-8 py-2 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
              />
              {videoId && (
                <button
                  onClick={() => {
                    setVideoId('');
                    handleApplyFilters(activity, '', minConfidence);
                  }}
                  className="absolute right-2.5 top-2.5 text-slate-500 hover:text-white"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Min Confidence Slider */}
          <div className="sm:col-span-2 lg:col-span-1">
            <div className="flex items-center justify-between text-xs font-medium text-slate-400 mb-1.5">
              <span className="flex items-center gap-1.5">
                <SlidersHorizontal className="w-3.5 h-3.5 text-amber-400" />
                Min Confidence
              </span>
              <span className="font-mono text-blue-400 font-bold">
                {minConfidence > 0 ? `${(minConfidence * 100).toFixed(0)}%` : 'Any'}
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="0.95"
              step="0.05"
              value={minConfidence}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setMinConfidence(val);
                handleApplyFilters(activity, videoId, val);
              }}
              className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          {/* Reset Filters */}
          <div className="flex items-end">
            <button
              onClick={handleResetFilters}
              className="w-full px-3.5 py-2 text-xs font-medium text-slate-400 hover:text-white bg-[#0F172A] hover:bg-slate-800 rounded-xl border border-slate-700/80 transition-all flex items-center justify-center gap-1.5"
            >
              <X className="w-3.5 h-3.5" />
              Reset All Filters
            </button>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Events Results */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 gap-3">
          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
          <span className="text-xs text-slate-400">Querying events from MongoDB...</span>
        </div>
      ) : eventsData.events.length === 0 ? (
        <EmptyState
          icon={ShieldAlert}
          title="No detected events match your criteria"
          description="Try lowering the minimum confidence slider or clearing active search filters."
          action={
            <button
              onClick={handleResetFilters}
              className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium"
            >
              Reset Filters
            </button>
          }
        />
      ) : viewMode === 'grid' ? (
        /* Grid View */
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {eventsData.events.map((ev) => (
            <div
              key={ev.event_id}
              className="bg-[#111827] border border-slate-800 rounded-2xl overflow-hidden hover:border-slate-700 transition-all flex flex-col group"
            >
              {/* Snapshot thumbnail */}
              <div
                onClick={() => setSelectedEvent(ev)}
                className="relative bg-slate-950 aspect-video overflow-hidden cursor-pointer"
              >
                <img
                  src={getEvidenceUrl(ev.evidence_snapshot)}
                  alt={ev.activity_label}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180"><rect fill="%230F172A" width="320" height="180"/><text fill="%2364748B" font-family="sans-serif" font-size="12" x="50%" y="50%" text-anchor="middle">No Snapshot</text></svg>';
                  }}
                />
                <div className="absolute top-2.5 left-2.5">
                  <ActivityBadge activity={ev.activity_label} />
                </div>
                <div className="absolute bottom-2.5 right-2.5 px-2 py-0.5 rounded bg-black/75 backdrop-blur-sm text-[11px] font-mono text-white">
                  {ev.start_time_formatted}
                </div>
              </div>

              {/* Card Body */}
              <div className="p-4 flex-1 flex flex-col justify-between space-y-3">
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">Confidence</span>
                    <span className="text-xs font-bold text-white">
                      {(ev.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-indigo-500 h-1.5 rounded-full"
                      style={{ width: `${ev.confidence * 100}%` }}
                    />
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
                  <span className="font-mono">
                    {ev.start_time_formatted} &rarr; {ev.end_time_formatted}
                  </span>
                  <button
                    onClick={() => setSelectedEvent(ev)}
                    className="text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1"
                  >
                    <Eye className="w-3.5 h-3.5" /> Enlarge
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Table View */
        <div className="bg-[#111827] border border-slate-800 rounded-2xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-[11px] font-semibold text-slate-400 uppercase tracking-wider bg-[#0F172A]/60">
                  <th className="py-3 px-4">Snapshot</th>
                  <th className="py-3 px-4">Activity Category</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Start Time</th>
                  <th className="py-3 px-4">End Time</th>
                  <th className="py-3 px-4">Video ID</th>
                  <th className="py-3 px-4 text-right">Evidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {eventsData.events.map((ev) => (
                  <tr key={ev.event_id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-4">
                      <div
                        onClick={() => setSelectedEvent(ev)}
                        className="w-14 h-9 rounded-lg overflow-hidden bg-slate-900 border border-slate-700/60 cursor-pointer hover:border-blue-500 transition-all"
                      >
                        <img
                          src={getEvidenceUrl(ev.evidence_snapshot)}
                          alt={ev.activity_label}
                          className="w-full h-full object-cover"
                          onError={(e) => (e.target.style.display = 'none')}
                        />
                      </div>
                    </td>
                    <td className="py-2.5 px-4">
                      <ActivityBadge activity={ev.activity_label} />
                    </td>
                    <td className="py-2.5 px-4 font-semibold text-slate-200">
                      {(ev.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-300">
                      {ev.start_time_formatted}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-300">
                      {ev.end_time_formatted}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-400 truncate max-w-[150px]" title={ev.video_id}>
                      {ev.video_id}
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      <button
                        onClick={() => setSelectedEvent(ev)}
                        className="px-2.5 py-1 text-[11px] font-medium text-blue-400 hover:text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 rounded-md border border-blue-500/20 transition-all inline-flex items-center gap-1"
                      >
                        <Eye className="w-3 h-3" /> View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-4 border-t border-slate-800 text-xs">
          <div className="text-slate-400">
            Showing <span className="font-semibold text-white">{page * pageSize + 1}</span> to{' '}
            <span className="font-semibold text-white">
              {Math.min((page + 1) * pageSize, eventsData.total)}
            </span>{' '}
            of <span className="font-semibold text-white">{eventsData.total}</span> events
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-1.5 rounded-lg bg-[#111827] border border-slate-800 hover:bg-slate-800 text-slate-300 disabled:opacity-40 transition-all"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="font-mono text-slate-300">
              Page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="p-1.5 rounded-lg bg-[#111827] border border-slate-800 hover:bg-slate-800 text-slate-300 disabled:opacity-40 transition-all"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Enlarge Evidence Modal */}
      {selectedEvent && (
        <EvidenceModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </div>
  );
};

export default EventsPage;
