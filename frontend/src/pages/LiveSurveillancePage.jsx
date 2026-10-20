import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Video,
  Play,
  Square,
  Plus,
  RefreshCw,
  Bell,
  Check,
  CheckCheck,
  Eye,
  ShieldAlert,
  Trash2,
  Layers,
  Radio,
  Clock,
} from 'lucide-react';
import { api, getLiveStreamUrl, getWebSocketUrl } from '../api/client';
import { ActivityBadge } from '../components/StatusBadge';
import AddCameraModal from '../components/AddCameraModal';
import EvidenceModal from '../components/EvidenceModal';

export const LiveSurveillancePage = () => {
  const [cameras, setCameras] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);

  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState({});
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [activeTab, setActiveTab] = useState('incidents'); // 'incidents' | 'notifications'
  const [wsConnected, setWsConnected] = useState(false);
  const [liveToast, setLiveToast] = useState(null);

  const ownerId = 'default_owner';
  const wsRef = useRef(null);

  // Fetch initial data
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const [camsRes, incsRes, notifsRes] = await Promise.all([
        api.listCameras(ownerId),
        api.listLiveIncidents({ ownerId, limit: 30 }),
        api.listLiveNotifications({ ownerId, limit: 30 }),
      ]);
      setCameras(camsRes.cameras || []);
      setIncidents(incsRes.incidents || []);
      setNotifications(notifsRes.notifications || []);
      setUnreadCount(notifsRes.unread_count || 0);
    } catch (err) {
      console.error('Failed to fetch live surveillance data:', err);
    } finally {
      setLoading(false);
    }
  }, [ownerId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // WebSocket Connection for instant real-time alerts
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    const connectWs = () => {
      try {
        const url = getWebSocketUrl(ownerId);
        ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          setWsConnected(true);
          console.log('[LiveSurveillance] WebSocket connected to real-time incident broker.');
        };

        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.type === 'NEW_INCIDENT' && data.notification) {
              const notif = data.notification;

              // Display instant toast
              setLiveToast(notif);
              setTimeout(() => setLiveToast((current) => (current?.id === notif.id ? null : current)), 8000);

              // Prepend to notifications and incidents feed
              setNotifications((prev) => [notif, ...prev]);
              setUnreadCount((prev) => prev + 1);

              // Prepend to live incidents feed
              const newInc = {
                id: notif.incident_id,
                camera_id: notif.camera_id,
                camera_name: notif.camera_name,
                location: notif.location,
                activity_label: notif.activity_label,
                confidence: notif.confidence,
                duration_seconds: 1.5,
                evidence_snapshot: notif.evidence_snapshot,
                is_acknowledged: false,
                created_at: notif.created_at,
              };
              setIncidents((prev) => [newInc, ...prev.filter((i) => i.id !== notif.incident_id)]);
            }
          } catch (e) {
            console.error('[LiveSurveillance] Error parsing WS message:', e);
          }
        };

        ws.onclose = () => {
          setWsConnected(false);
          // Auto-reconnect after 4s
          reconnectTimeout = setTimeout(connectWs, 4000);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch (err) {
        console.error('[LiveSurveillance] WS connection error:', err);
        reconnectTimeout = setTimeout(connectWs, 4000);
      }
    };

    connectWs();

    // Periodic ping to keep WS connection alive
    const pingInterval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 20000);

    return () => {
      clearInterval(pingInterval);
      clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, [ownerId]);

  // Camera start / stop actions
  const handleToggleMonitoring = async (cam) => {
    setActionLoading((prev) => ({ ...prev, [cam.id]: true }));
    try {
      if (cam.is_monitored) {
        await api.stopCamera(cam.id);
      } else {
        await api.startCamera(cam.id);
      }
      // Refresh camera state
      const updated = await api.listCameras(ownerId);
      setCameras(updated.cameras || []);
    } catch (err) {
      alert(err.response?.data?.detail || err.message || 'Operation failed');
    } finally {
      setActionLoading((prev) => ({ ...prev, [cam.id]: false }));
    }
  };

  const handleDeleteCamera = async (camId) => {
    if (!window.confirm('Are you sure you want to delete this camera? Live monitoring will cease.')) return;
    try {
      await api.deleteCamera(camId);
      setCameras((prev) => prev.filter((c) => c.id !== camId));
    } catch (err) {
      alert(err.response?.data?.detail || err.message || 'Failed to delete camera');
    }
  };

  const handleAcknowledgeIncident = async (incidentId) => {
    try {
      const res = await api.acknowledgeLiveIncident(incidentId, ownerId);
      setIncidents((prev) => prev.map((inc) => (inc.id === incidentId ? res : inc)));
      setNotifications((prev) =>
        prev.map((n) => (n.incident_id === incidentId ? { ...n, is_acknowledged: true, is_read: true } : n))
      );
    } catch (err) {
      alert(err.response?.data?.detail || err.message || 'Failed to acknowledge incident');
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await api.markAllNotificationsRead(ownerId);
      setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch (err) {
      console.error('Failed to mark all read:', err);
    }
  };

  const activeCamerasCount = cameras.filter((c) => c.is_monitored).length;

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Toast Alert Banner for Live Detections */}
      {liveToast && (
        <div className="fixed top-6 right-6 z-50 max-w-md w-full bg-rose-950/90 border border-rose-500/50 rounded-2xl p-4 shadow-2xl backdrop-blur-md animate-in slide-in-from-top-4 duration-300">
          <div className="flex items-start gap-3">
            <div className="p-2.5 rounded-xl bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
                  Live Intrusion Detected
                </span>
                <span className="text-[11px] text-slate-400">Just now</span>
              </div>
              <h4 className="text-sm font-semibold text-white mt-1">
                {liveToast.activity_label} detected on {liveToast.camera_name}
              </h4>
              <p className="text-xs text-slate-300 mt-0.5">
                Location: {liveToast.location || 'Zone'} • Confidence: {(liveToast.confidence * 100).toFixed(1)}%
              </p>
              {liveToast.evidence_snapshot && (
                <div className="mt-2.5 flex items-center gap-2">
                  <button
                    onClick={() =>
                      setSelectedEvent({
                        activity_label: liveToast.activity_label,
                        confidence: liveToast.confidence,
                        evidence_snapshot: liveToast.evidence_snapshot,
                        start_time_formatted: 'Live Real-Time',
                        video_id: liveToast.camera_id,
                      })
                    }
                    className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 text-xs font-medium border border-rose-500/30 transition-colors"
                  >
                    <Eye className="w-3.5 h-3.5" />
                    Inspect Snapshot
                  </button>
                  <button
                    onClick={() => handleAcknowledgeIncident(liveToast.incident_id)}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Acknowledge
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold tracking-tight text-white">Live CCTV Surveillance</h1>
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                wsConnected
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                  : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
              {wsConnected ? 'Real-Time Connected' : 'Reconnecting...'}
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Real-time multi-camera stream processing with shared 3D CNN inference and instant owner notifications.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-[#1E293B] hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => setAddModalOpen(true)}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20 transition-all"
          >
            <Plus className="w-4 h-4" />
            Register Camera
          </button>
        </div>
      </div>

      {/* Stats Counter Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Total Cameras</span>
            <Video className="w-4 h-4 text-blue-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{cameras.length}</p>
        </div>
        <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Active Streams</span>
            <Radio className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-emerald-400 mt-2">{activeCamerasCount}</p>
        </div>
        <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Live Incidents</span>
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          </div>
          <p className="text-2xl font-bold text-rose-400 mt-2">{incidents.length}</p>
        </div>
        <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">Unread Alerts</span>
            <Bell className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-2xl font-bold text-amber-400 mt-2">{unreadCount}</p>
        </div>
      </div>

      {/* Main Layout: Camera Feeds on Left, Live Incidents / Notifications on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Registered Cameras & Feeds */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-400" />
              Camera Feeds
              <span className="text-xs font-normal text-slate-400">({cameras.length} registered)</span>
            </h2>
          </div>

          {cameras.length === 0 ? (
            <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-12 text-center">
              <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center justify-center mx-auto mb-3">
                <Video className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-white">No cameras registered yet</h3>
              <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1 mb-4">
                Register an RTSP IP camera, local USB webcam, or synthetic simulator to start real-time monitoring.
              </p>
              <button
                onClick={() => setAddModalOpen(true)}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20 transition-all"
              >
                <Plus className="w-4 h-4" />
                Register First Camera
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {cameras.map((cam) => {
                const isOnline = cam.status === 'online';
                const isMonitored = cam.is_monitored;
                const isBusy = actionLoading[cam.id];

                return (
                  <div
                    key={cam.id}
                    className="bg-[#0F172A] border border-slate-800/80 rounded-2xl overflow-hidden flex flex-col hover:border-slate-700 transition-all shadow-lg"
                  >
                    {/* Camera Header */}
                    <div className="p-3.5 border-b border-slate-800/80 flex items-center justify-between">
                      <div className="min-w-0 pr-2">
                        <h4 className="text-sm font-semibold text-white truncate">{cam.name}</h4>
                        <p className="text-[11px] text-slate-400 truncate">
                          {cam.location || 'Zone'} • <span className="font-mono text-slate-500">{cam.stream_type}</span>
                        </p>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider border ${
                            isOnline
                              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                              : isMonitored
                              ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                              : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              isOnline ? 'bg-emerald-500 animate-pulse' : isMonitored ? 'bg-amber-500' : 'bg-slate-500'
                            }`}
                          />
                          {cam.status}
                        </span>
                        <button
                          onClick={() => handleDeleteCamera(cam.id)}
                          className="p-1 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                          title="Delete camera"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* Stream Preview Box */}
                    <div className="relative aspect-video bg-black/60 flex items-center justify-center overflow-hidden">
                      {isMonitored ? (
                        <img
                          src={getLiveStreamUrl(cam.id)}
                          alt={cam.name}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            // If stream momentarily offline, don't break layout
                            e.target.style.display = 'none';
                          }}
                        />
                      ) : (
                        <div className="text-center p-6">
                          <Video className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                          <p className="text-xs font-medium text-slate-400">Monitoring Inactive</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Click Start to stream and detect</p>
                        </div>
                      )}

                      {/* On-Screen Stream Badges */}
                      {isMonitored && (
                        <div className="absolute top-2 left-2 flex items-center gap-1 px-2 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[10px] text-emerald-400 font-mono border border-emerald-500/30">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
                          LIVE REC
                        </div>
                      )}
                      {cam.fps && (
                        <div className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[10px] text-slate-300 font-mono">
                          {cam.fps} FPS
                        </div>
                      )}
                    </div>

                    {/* Camera Footer Controls */}
                    <div className="p-3 bg-[#0B0F17]/50 border-t border-slate-800/80 flex items-center justify-between text-xs">
                      <div className="text-[11px] text-slate-400 font-mono">
                        Threshold: {(cam.confidence_threshold * 100).toFixed(0)}%
                      </div>

                      <button
                        onClick={() => handleToggleMonitoring(cam)}
                        disabled={isBusy}
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                          isMonitored
                            ? 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30'
                            : 'bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        } disabled:opacity-50`}
                      >
                        {isBusy ? (
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        ) : isMonitored ? (
                          <>
                            <Square className="w-3 h-3 fill-current" />
                            Stop Monitoring
                          </>
                        ) : (
                          <>
                            <Play className="w-3 h-3 fill-current" />
                            Start Monitoring
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right 1 Col: Incidents & Alerts Tabbed Feed */}
        <div className="space-y-4">
          {/* Tabs */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActiveTab('incidents')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-colors ${
                  activeTab === 'incidents'
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                Live Incidents ({incidents.length})
              </button>
              <button
                onClick={() => setActiveTab('notifications')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-colors relative ${
                  activeTab === 'notifications'
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Bell className="w-3.5 h-3.5" />
                Alerts
                {unreadCount > 0 && (
                  <span className="w-4 h-4 rounded-full bg-rose-500 text-white text-[10px] font-bold flex items-center justify-center">
                    {unreadCount}
                  </span>
                )}
              </button>
            </div>

            {activeTab === 'notifications' && unreadCount > 0 && (
              <button
                onClick={handleMarkAllRead}
                className="text-[11px] text-blue-400 hover:text-blue-300 flex items-center gap-1"
              >
                <CheckCheck className="w-3 h-3" />
                Mark all read
              </button>
            )}
          </div>

          {/* Tab 1: Live Incidents */}
          {activeTab === 'incidents' && (
            <div className="space-y-3 max-h-[680px] overflow-y-auto pr-1">
              {incidents.length === 0 ? (
                <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-8 text-center text-slate-400">
                  <ShieldAlert className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-xs font-medium">No live incidents detected</p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    When suspicious activity occurs on monitored cameras, evidence will appear here in real-time.
                  </p>
                </div>
              ) : (
                incidents.map((inc) => (
                  <div
                    key={inc.id}
                    className={`bg-[#0F172A] border rounded-2xl p-3.5 transition-all ${
                      inc.is_acknowledged
                        ? 'border-slate-800/80 opacity-75'
                        : 'border-rose-500/30 shadow-lg shadow-rose-950/20'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <ActivityBadge activity={inc.activity_label} />
                      <span className="text-xs font-mono font-semibold text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">
                        {(inc.confidence * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="text-xs text-slate-300 font-medium">
                      {inc.camera_name || 'Camera'} • <span className="text-slate-400">{inc.location || 'Zone'}</span>
                    </div>

                    <div className="flex items-center gap-3 text-[11px] text-slate-400 mt-1">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(inc.created_at).toLocaleTimeString()}
                      </span>
                      {inc.duration_seconds > 0 && (
                        <span>Duration: {inc.duration_seconds.toFixed(1)}s</span>
                      )}
                    </div>

                    {/* Snapshot & Actions */}
                    <div className="mt-3 flex items-center justify-between pt-2.5 border-t border-slate-800">
                      {inc.evidence_snapshot ? (
                        <button
                          onClick={() =>
                            setSelectedEvent({
                              activity_label: inc.activity_label,
                              confidence: inc.confidence,
                              evidence_snapshot: inc.evidence_snapshot,
                              start_time_formatted: new Date(inc.created_at).toLocaleTimeString(),
                              video_id: inc.camera_id,
                            })
                          }
                          className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 font-medium"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          View Snapshot
                        </button>
                      ) : (
                        <span className="text-[11px] text-slate-500 italic">No snapshot</span>
                      )}

                      {!inc.is_acknowledged ? (
                        <button
                          onClick={() => handleAcknowledgeIncident(inc.id)}
                          className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium border border-slate-700 transition-colors"
                        >
                          <Check className="w-3 h-3" />
                          Acknowledge
                        </button>
                      ) : (
                        <span className="flex items-center gap-1 text-[11px] text-emerald-400">
                          <CheckCheck className="w-3.5 h-3.5" />
                          Acknowledged
                        </span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab 2: Notifications */}
          {activeTab === 'notifications' && (
            <div className="space-y-2.5 max-h-[680px] overflow-y-auto pr-1">
              {notifications.length === 0 ? (
                <div className="bg-[#0F172A] border border-slate-800/80 rounded-2xl p-8 text-center text-slate-400">
                  <Bell className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-xs font-medium">No alerts right now</p>
                </div>
              ) : (
                notifications.map((n) => (
                  <div
                    key={n.id}
                    className={`bg-[#0F172A] border rounded-xl p-3 text-xs transition-colors ${
                      n.is_read ? 'border-slate-800/80' : 'border-blue-500/30 bg-blue-950/10'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-white">{n.activity_label}</span>
                      <span className="text-[10px] text-slate-500">
                        {new Date(n.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-slate-300 text-[11px] mt-0.5">
                      Camera: {n.camera_name} ({n.location || 'Zone'}) • {(n.confidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {/* Add Camera Modal */}
      <AddCameraModal
        isOpen={addModalOpen}
        onClose={() => setAddModalOpen(false)}
        onCameraAdded={(newCam) => setCameras((prev) => [newCam, ...prev])}
      />

      {/* Evidence Enlargement Modal */}
      {selectedEvent && (
        <EvidenceModal event={selectedEvent} onClose={() => setSelectedEvent(null)} />
      )}
    </div>
  );
};

export default LiveSurveillancePage;
