import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  UploadCloud,
  FileVideo,
  Play,
  AlertCircle,
  Clock,
  Sliders,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Sparkles
} from 'lucide-react';
import { api } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';

const ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'];

export const VideoAnalysisPage = () => {

  // Upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState(null);

  // Ingestion & Processing state
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState(null);
  const [processError, setProcessError] = useState(null);

  // Parameters
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.40);
  const [windowSec, setWindowSec] = useState(2.0);
  const [strideSec, setStrideSec] = useState(1.0);

  // Past videos
  const [recentVideos, setRecentVideos] = useState([]);
  const [loadingVideos, setLoadingVideos] = useState(false);

  const fileInputRef = useRef(null);
  const pollTimerRef = useRef(null);

  const fetchRecentVideos = async () => {
    try {
      const res = await api.getVideos(15, 0);
      setRecentVideos(res.videos || []);
    } catch (err) {
      console.error('Failed to fetch recent videos:', err);
    } finally {
      setLoadingVideos(false);
    }
  };

  useEffect(() => {
    fetchRecentVideos();
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  // Format file size
  const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const validateAndSelectFile = (file) => {
    setUploadError(null);
    if (!file) return;

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setUploadError(`Invalid file format '${ext}'. Allowed formats: ${ALLOWED_EXTENSIONS.join(', ')}`);
      return;
    }

    if (file.size === 0) {
      setUploadError('Uploaded file cannot be empty (0 bytes).');
      return;
    }

    setSelectedFile(file);
    setUploadedVideo(null);
    setProcessStatus(null);
    setProcessError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  // Upload video
  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    try {
      const data = await api.uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });
      setUploadedVideo(data);
      fetchRecentVideos();
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Video upload failed.';
      setUploadError(msg);
    } finally {
      setUploading(false);
    }
  };

  // Start background processing & poll status
  const startProcessing = async (targetVideoId) => {
    const videoId = targetVideoId || uploadedVideo?.video_id;
    if (!videoId) return;

    setIsProcessing(true);
    setProcessError(null);
    setProcessStatus({ status: 'processing', events_count: 0 });

    try {
      await api.processVideo(videoId, {
        confidenceThreshold: parseFloat(confidenceThreshold),
        windowSec: parseFloat(windowSec),
        strideSec: parseFloat(strideSec),
      });

      // Begin polling every 2 seconds
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);

      pollTimerRef.current = setInterval(async () => {
        try {
          const statusData = await api.getVideoStatus(videoId);
          setProcessStatus(statusData);

          if (statusData.status === 'completed' || statusData.status === 'failed') {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setIsProcessing(false);
            fetchRecentVideos();
          }
        } catch (pollErr) {
          console.error('Status poll error:', pollErr);
        }
      }, 2000);
    } catch (err) {
      setIsProcessing(false);
      const msg = err.response?.data?.detail || err.message || 'Failed to start processing.';
      setProcessError(msg);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-3">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          CCTV Video Analysis
          <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
            r2plus1d_18
          </span>
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Ingest video recordings, configure detection thresholds, and run GPU-accelerated action recognition.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload & Config Area (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* File Upload Dropzone */}
          <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
            <h3 className="text-sm font-semibold text-slate-100 mb-3 flex items-center gap-2">
              <UploadCloud className="w-4 h-4 text-blue-400" />
              1. Ingest Video Footage
            </h3>

            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : selectedFile
                  ? 'border-slate-700 bg-slate-900/60'
                  : 'border-slate-700/80 hover:border-slate-600 bg-slate-900/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={ALLOWED_EXTENSIONS.join(',')}
                onChange={handleFileInput}
                className="hidden"
              />

              <div className="p-3 bg-slate-800 rounded-2xl text-slate-300 w-12 h-12 mx-auto mb-3 flex items-center justify-center border border-slate-700">
                <FileVideo className="w-6 h-6 text-blue-400" />
              </div>

              {selectedFile ? (
                <div>
                  <div className="text-sm font-semibold text-white">{selectedFile.name}</div>
                  <div className="text-xs text-slate-400 mt-1">
                    {formatBytes(selectedFile.size)} • Click or drop another file to replace
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-sm font-semibold text-slate-200">
                    Click to select or drag and drop CCTV video
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    Supported formats: MP4, AVI, MOV, MKV
                  </div>
                </div>
              )}
            </div>

            {/* Upload Error Banner */}
            {uploadError && (
              <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            {/* Upload Progress Bar */}
            {uploading && (
              <div className="mt-4 space-y-1.5">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Uploading to server...</span>
                  <span className="font-semibold text-blue-400">{uploadProgress}%</span>
                </div>
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-200"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div className="mt-5 flex items-center justify-end gap-3">
              {selectedFile && !uploadedVideo && (
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-blue-600/20 transition-all disabled:opacity-50 flex items-center gap-2"
                >
                  {uploading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <UploadCloud className="w-3.5 h-3.5" />
                      Upload to Storage
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Model Inference Configuration */}
          <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
            <h3 className="text-sm font-semibold text-slate-100 mb-4 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              2. Action Recognition Parameters
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              {/* Confidence Threshold */}
              <div>
                <div className="flex items-center justify-between text-xs font-medium text-slate-300 mb-1.5">
                  <span>Confidence Threshold</span>
                  <span className="font-mono text-blue-400 font-bold">
                    {(confidenceThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(e.target.value)}
                  className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
                <div className="text-[11px] text-slate-500 mt-1">
                  Filters detections below this probability.
                </div>
              </div>

              {/* Window Size */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">
                  Sliding Window (Seconds)
                </label>
                <input
                  type="number"
                  min="0.5"
                  max="10.0"
                  step="0.5"
                  value={windowSec}
                  onChange={(e) => setWindowSec(e.target.value)}
                  className="w-full bg-[#0F172A] border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
                />
                <div className="text-[11px] text-slate-500 mt-1">
                  Temporal clip duration for 3D CNN.
                </div>
              </div>

              {/* Stride */}
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">
                  Stride Step (Seconds)
                </label>
                <input
                  type="number"
                  min="0.2"
                  max="5.0"
                  step="0.2"
                  value={strideSec}
                  onChange={(e) => setStrideSec(e.target.value)}
                  className="w-full bg-[#0F172A] border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500 font-mono"
                />
                <div className="text-[11px] text-slate-500 mt-1">
                  Sliding window advancement step.
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Live Processing Status Tracker (1 col) */}
        <div className="space-y-6">
          <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
            <h3 className="text-sm font-semibold text-slate-100 mb-4 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              3. Processing Engine
            </h3>

            {uploadedVideo ? (
              <div className="space-y-4">
                {/* Uploaded Video Card */}
                <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-3.5 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-white truncate max-w-[180px]">
                      {uploadedVideo.filename}
                    </span>
                    <StatusBadge status={uploadedVideo.status} />
                  </div>
                  <div className="text-[11px] text-slate-400 grid grid-cols-2 gap-1 font-mono">
                    <div>Duration: {uploadedVideo.duration || '—'}s</div>
                    <div>FPS: {uploadedVideo.fps || '30'}</div>
                    <div className="col-span-2 truncate text-slate-500">ID: {uploadedVideo.video_id}</div>
                  </div>
                </div>

                {/* Trigger Button */}
                <button
                  onClick={() => startProcessing()}
                  disabled={isProcessing}
                  className="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl text-xs font-semibold shadow-lg shadow-blue-600/20 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Analyzing Video...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-current" />
                      Start AI Action Analysis
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="py-8 text-center text-xs text-slate-500">
                Upload a video on the left to begin action recognition analysis.
              </div>
            )}

            {/* Process Error */}
            {processError && (
              <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{processError}</span>
              </div>
            )}

            {/* Active Status Display */}
            {processStatus && (
              <div className="mt-5 p-4 rounded-xl bg-[#0F172A] border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300">Analysis Status</span>
                  <StatusBadge status={processStatus.status} />
                </div>

                {processStatus.status === 'processing' && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs text-amber-400">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Extracting spatiotemporal clips...</span>
                    </div>
                    <p className="text-[11px] text-slate-500">
                      Analyzing frames with R(2+1)D model on GPU. This usually takes 5-15 seconds.
                    </p>
                  </div>
                )}

                {processStatus.status === 'completed' && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-xs text-emerald-400 font-medium">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Analysis Completed Successfully!</span>
                    </div>

                    <div className="text-xs text-slate-300">
                      Detected Suspicious Events: <span className="font-bold text-white">{processStatus.events_count}</span>
                    </div>

                    <Link
                      to={`/events?videoId=${processStatus.video_id}`}
                      className="block text-center py-2 px-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-all"
                    >
                      View Detected Events ({processStatus.events_count}) &rarr;
                    </Link>
                  </div>
                )}

                {processStatus.status === 'failed' && (
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-xs text-rose-400 font-medium">
                      <XCircle className="w-4 h-4" />
                      <span>Analysis Failed</span>
                    </div>
                    {processStatus.error_message && (
                      <p className="text-[11px] text-rose-300/80 font-mono">
                        {processStatus.error_message}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Past Ingested Recordings */}
      <div className="bg-[#111827] border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Clock className="w-4 h-4 text-slate-400" />
            Ingested Video Archive
          </h3>
          <button
            onClick={() => {
              setLoadingVideos(true);
              fetchRecentVideos();
            }}
            disabled={loadingVideos}
            className="text-xs text-slate-400 hover:text-white flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingVideos ? 'animate-spin text-blue-400' : ''}`} />
            Refresh List
          </button>
        </div>

        {recentVideos.length === 0 ? (
          <div className="py-6 text-center text-xs text-slate-500">
            No previous video records found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                  <th className="py-2.5 px-3">Filename</th>
                  <th className="py-2.5 px-3">Duration</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Events Detected</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {recentVideos.map((vid) => (
                  <tr key={vid.video_id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-3 font-medium text-slate-200">
                      <div className="truncate max-w-[240px]" title={vid.filename}>{vid.filename}</div>
                      <div className="text-[10px] text-slate-500 font-mono truncate max-w-[240px]">{vid.video_id}</div>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 font-mono">
                      {vid.duration ? `${vid.duration}s` : '—'}
                    </td>
                    <td className="py-2.5 px-3">
                      <StatusBadge status={vid.status} />
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">
                      {vid.events_count}
                    </td>
                    <td className="py-2.5 px-3 text-right space-x-2">
                      <button
                        onClick={() => startProcessing(vid.video_id)}
                        disabled={isProcessing}
                        className="px-2.5 py-1 text-[11px] font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-md border border-slate-700 transition-all disabled:opacity-50"
                      >
                        Re-Analyze
                      </button>
                      <Link
                        to={`/events?videoId=${vid.video_id}`}
                        className="px-2.5 py-1 text-[11px] font-medium bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 rounded-md border border-blue-500/20 transition-all inline-block"
                      >
                        View Events
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoAnalysisPage;
