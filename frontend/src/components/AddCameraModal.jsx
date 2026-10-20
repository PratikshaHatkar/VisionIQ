import React, { useState } from 'react';
import { X, Video, Sliders, CheckCircle2, AlertCircle } from 'lucide-react';
import { api } from '../api/client';

export const AddCameraModal = ({ isOpen, onClose, onCameraAdded }) => {
  const [formData, setFormData] = useState({
    name: '',
    location: '',
    owner_id: 'default_owner',
    stream_type: 'synthetic',
    stream_url: 'synthetic:0',
    confidence_threshold: 0.50,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleStreamTypeChange = (type) => {
    let defaultUrl = '';
    if (type === 'synthetic') defaultUrl = 'synthetic:0';
    else if (type === 'webcam') defaultUrl = '0';
    else if (type === 'rtsp') defaultUrl = 'rtsp://admin:password@192.168.1.100:554/live';
    else if (type === 'test_video') defaultUrl = 'sample_cctv.mp4';

    setFormData((prev) => ({
      ...prev,
      stream_type: type,
      stream_url: defaultUrl,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.name.trim() || !formData.stream_url.trim()) {
      setError('Camera name and stream URL are required.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const created = await api.registerCamera(formData);
      onCameraAdded(created);
      onClose();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to register camera');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#0F172A] border border-slate-800 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/80">
          <div className="flex items-center gap-2 text-white font-semibold text-lg">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Video className="w-5 h-5" />
            </div>
            Register CCTV Camera
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Camera Name & Location */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Camera Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="e.g. North Gate CCTV"
                required
                className="w-full bg-[#1E293B] border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Location / Zone
              </label>
              <input
                type="text"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                placeholder="e.g. Zone A - Entrance"
                className="w-full bg-[#1E293B] border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
          </div>

          {/* Stream Type Selection */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Stream Type
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {[
                { id: 'synthetic', label: 'Synthetic Sim' },
                { id: 'rtsp', label: 'RTSP IP Cam' },
                { id: 'webcam', label: 'Local Webcam' },
                { id: 'test_video', label: 'Test Video Loop' },
              ].map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => handleStreamTypeChange(opt.id)}
                  className={`px-3 py-2 rounded-xl text-xs font-medium border transition-all text-center ${
                    formData.stream_type === opt.id
                      ? 'bg-blue-600/20 border-blue-500 text-blue-400 font-semibold'
                      : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Stream URL / Device */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Stream Source / Endpoint *
            </label>
            <input
              type="text"
              value={formData.stream_url}
              onChange={(e) => setFormData({ ...formData, stream_url: e.target.value })}
              placeholder={
                formData.stream_type === 'rtsp'
                  ? 'rtsp://user:pass@192.168.1.100:554/live'
                  : formData.stream_type === 'webcam'
                  ? '0 (device index)'
                  : 'synthetic:0'
              }
              required
              className="w-full bg-[#1E293B] border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white font-mono placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
            />
            <p className="text-[11px] text-slate-400 mt-1">
              Credentials in RTSP URLs are masked and protected across all dashboard responses.
            </p>
          </div>

          {/* Confidence Threshold Slider */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-medium text-slate-300 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-blue-400" />
                Alert Sensitivity Threshold
              </label>
              <span className="text-xs font-mono font-semibold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">
                {Math.round(formData.confidence_threshold * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.20"
              max="0.95"
              step="0.05"
              value={formData.confidence_threshold}
              onChange={(e) =>
                setFormData({ ...formData, confidence_threshold: parseFloat(e.target.value) })
              }
              className="w-full accent-blue-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
              <span>More Alerts (20%)</span>
              <span>Balanced (50%)</span>
              <span>Strict / Verified (95%)</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-800">
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-1.5 px-5 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>Registering...</>
              ) : (
                <>
                  <CheckCircle2 className="w-4 h-4" />
                  Save Camera
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddCameraModal;
