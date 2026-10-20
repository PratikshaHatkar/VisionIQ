import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  ShieldAlert,
  Video,
  Cpu,
  Database,
  RefreshCw,
  PlusCircle,
  Clock,
  ArrowRight,
  AlertCircle,
  Eye,
} from 'lucide-react';
import { api, getEvidenceUrl } from '../api/client';
import StatsCard from '../components/StatsCard';
import { StatusBadge, ActivityBadge } from '../components/StatusBadge';
import EvidenceModal from '../components/EvidenceModal';
import EmptyState from '../components/EmptyState';

export const OverviewPage = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [healthData, setHealthData] = useState(null);
  const [eventsData, setEventsData] = useState({ total: 0, events: [] });
  const [videosData, setVideosData] = useState({ total: 0, videos: [] });
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [health, events, videos] = await Promise.allSettled([
        api.getHealth(),
        api.getEvents({ limit: 50 }),
        api.getVideos(10, 0),
      ]);
      setError(null);

      if (health.status === 'fulfilled') {
        setHealthData(health.value);
      } else {
        throw new Error('Could not reach VisionIQ FastAPI backend. Please ensure the server is running on http://127.0.0.1:8000.');
      }

      if (events.status === 'fulfilled') {
        setEventsData(events.value);
      }

      if (videos.status === 'fulfilled') {
        setVideosData(videos.value);
      }
    } catch (err) {
      setError(err.message || 'Failed to load surveillance dashboard data.');
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchData();
  };

  // Derive genuine activity distribution counts from real returned events
  const activityDistribution = React.useMemo(() => {
    const counts = {};
    (eventsData.events || []).forEach((ev) => {
      const label = ev.activity_label || 'Other';
      counts[label] = (counts[label] || 0) + 1;
    });

    const total = eventsData.events?.length || 0;
    return Object.entries(counts).map(([label, count]) => ({
      label,
      count,
      percentage: total > 0 ? Math.round((count / total) * 100) : 0,
    })).sort((a, b) => b.count - a.count);
  }, [eventsData.events]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
        <span className="text-sm text-slate-400">Loading surveillance dashboard metrics...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Header & Quick Action */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            Surveillance Overview
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Live Monitor
            </span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Real-time automated suspicious activity recognition and event persistence.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#111827] text-slate-300 border border-slate-700/80 hover:bg-slate-800 hover:text-white transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-blue-400' : ''}`} />
            Refresh
          </button>
          <Link
            to="/analysis"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 text-white shadow-lg shadow-blue-600/20 hover:bg-blue-500 transition-all"
          >
            <PlusCircle className="w-4 h-4" />
            Analyze Video
          </Link>
        </div>
      </div>

      {/* Error Alert if backend unreachable */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 flex items-start justify-between gap-3">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <div className="text-sm font-semibold">Backend Connection Issue</div>
              <div className="text-xs text-rose-300/80 mt-0.5">{error}</div>
            </div>
          </div>
          <button
            onClick={fetchData}
            className="px-3 py-1 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 rounded-lg text-xs font-medium border border-rose-500/40"
          >
            Retry
          </button>
        </div>
      )}

      {/* Key Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Suspicious Events"
          value={eventsData.total}
          subtitle={`${eventsData.events?.length || 0} recent alerts loaded`}
          icon={ShieldAlert}
          accentColor="rose"
          badge={eventsData.total > 0 ? 'Active Alerts' : 'No Alerts'}
        />

        <StatsCard
          title="Uploaded Videos"
          value={videosData.total}
          subtitle="Processed CCTV recordings"
          icon={Video}
          accentColor="blue"
        />

        <StatsCard
          title="AI Model Engine"
          value={healthData?.model?.name || 'r2plus1d_18'}
          subtitle={`Running on ${healthData?.model?.device?.toUpperCase() || 'GPU'}`}
          icon={Cpu}
          accentColor="indigo"
          badge={healthData?.model?.loaded ? 'Loaded' : 'Ready'}
        />

        <StatsCard
          title="Database Persistence"
          value={healthData?.database_engine === 'mongodb' ? 'MongoDB' : 'SQLite Fallback'}
          subtitle={healthData?.database_details || 'Connected'}
          icon={Database}
          accentColor="emerald"
          badge={healthData?.database === 'connected' ? 'Healthy' : 'Warning'}
        />
      </div>

      {/* Main Grid: Activity Distribution & Recent Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Distribution Breakdown */}
        <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                Detected Activity Breakdown
              </h3>
              <span className="text-xs text-slate-400">
                {eventsData.events?.length || 0} Events
              </span>
            </div>

            {activityDistribution.length === 0 ? (
              <div className="py-8 text-center text-xs text-slate-500">
                No activity distribution available yet. Upload and analyze a CCTV video to detect suspicious events.
              </div>
            ) : (
              <div className="space-y-3.5">
                {activityDistribution.map((item) => (
                  <div key={item.label} className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-300">{item.label}</span>
                      <span className="text-slate-400">
                        {item.count} ({item.percentage}%)
                      </span>
                    </div>
                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800/80">
            <Link
              to="/events"
              className="text-xs text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-1.5"
            >
              Explore all detected event logs <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {/* Recent Suspicious Events (2 cols on lg) */}
        <div className="lg:col-span-2 bg-[#111827] border border-slate-800 rounded-2xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-400" />
              Recent Suspicious Detections
            </h3>
            <Link
              to="/events"
              className="text-xs text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-1"
            >
              View all ({eventsData.total}) <ArrowRight className="w-3 h-3" />
            </Link>
          </div>

          {(!eventsData.events || eventsData.events.length === 0) ? (
            <EmptyState
              title="No suspicious events detected yet"
              description="Upload a CCTV video under Video Analysis to run spatiotemporal action recognition."
              action={
                <Link
                  to="/analysis"
                  className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium"
                >
                  <PlusCircle className="w-3.5 h-3.5" /> Start First Analysis
                </Link>
              }
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    <th className="py-2.5 px-3">Evidence</th>
                    <th className="py-2.5 px-3">Activity</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">Time Range</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-xs">
                  {eventsData.events.slice(0, 5).map((ev) => (
                    <tr key={ev.event_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2.5 px-3">
                        <div
                          onClick={() => setSelectedEvent(ev)}
                          className="w-14 h-9 rounded-lg overflow-hidden bg-slate-900 border border-slate-700/60 cursor-pointer hover:border-blue-500 transition-all relative group"
                        >
                          <img
                            src={getEvidenceUrl(ev.evidence_snapshot)}
                            alt={ev.activity_label}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              e.target.style.display = 'none';
                            }}
                          />
                          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                            <Eye className="w-3.5 h-3.5 text-white" />
                          </div>
                        </div>
                      </td>
                      <td className="py-2.5 px-3">
                        <ActivityBadge activity={ev.activity_label} />
                      </td>
                      <td className="py-2.5 px-3 font-medium text-slate-200">
                        {(ev.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-400">
                        {ev.start_time_formatted} - {ev.end_time_formatted}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <button
                          onClick={() => setSelectedEvent(ev)}
                          className="px-2.5 py-1 text-[11px] font-medium text-blue-400 hover:text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 rounded-md border border-blue-500/20 transition-all"
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Recent Processed Videos Table */}
      <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Video className="w-4 h-4 text-emerald-400" />
            Recently Ingested CCTV Footage
          </h3>
          <Link
            to="/analysis"
            className="text-xs text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-1"
          >
            Upload New Video <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        {(!videosData.videos || videosData.videos.length === 0) ? (
          <EmptyState
            title="No videos ingested yet"
            description="Use the Video Analysis page to upload MP4/AVI surveillance recordings."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  <th className="py-2.5 px-3">Filename</th>
                  <th className="py-2.5 px-3">Resolution</th>
                  <th className="py-2.5 px-3">Duration</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Events Found</th>
                  <th className="py-2.5 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-xs">
                {videosData.videos.slice(0, 6).map((vid) => (
                  <tr key={vid.video_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-3 font-medium text-slate-200 flex items-center gap-2">
                      <Video className="w-3.5 h-3.5 text-slate-400" />
                      <span className="truncate max-w-[200px]" title={vid.filename}>{vid.filename}</span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 font-mono">
                      {vid.duration ? `${vid.duration}s` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 font-mono">
                      {vid.fps ? `${vid.fps} FPS` : '30 FPS'}
                    </td>
                    <td className="py-2.5 px-3">
                      <StatusBadge status={vid.status} />
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">
                      {vid.events_count || 0}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <Link
                        to={`/events?videoId=${vid.video_id}`}
                        className="px-2.5 py-1 text-[11px] font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-md border border-slate-700 transition-all inline-block"
                      >
                        Filter Events
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Evidence Snapshot Inspection Modal */}
      {selectedEvent && (
        <EvidenceModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </div>
  );
};

export default OverviewPage;
