import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000, // 30s timeout for video operations
  headers: {
    'Accept': 'application/json',
  },
});

// Helper to get fully qualified evidence image URL
export const getEvidenceUrl = (snapshotPath) => {
  if (!snapshotPath) return null;
  if (snapshotPath.startsWith('http://') || snapshotPath.startsWith('https://')) {
    return snapshotPath;
  }
  const cleanPath = snapshotPath.startsWith('/') ? snapshotPath : `/${snapshotPath}`;
  return `${BASE_URL}${cleanPath}`;
};

// API Services
export const api = {
  // 1. Health & System Diagnostic
  async getHealth() {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // 2. Video Upload
  async uploadVideo(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await apiClient.post('/api/videos/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
    return res.data;
  },

  // 3. Process Video
  async processVideo(videoId, { confidenceThreshold = 0.40, windowSec = 2.0, strideSec = 1.0 } = {}) {
    const res = await apiClient.post(`/api/videos/${videoId}/process`, null, {
      params: {
        confidence_threshold: confidenceThreshold,
        window_sec: windowSec,
        stride_sec: strideSec,
      },
    });
    return res.data;
  },

  // 4. Video Status
  async getVideoStatus(videoId) {
    const res = await apiClient.get(`/api/videos/${videoId}/status`);
    return res.data;
  },

  // 5. List Videos
  async getVideos(limit = 50, offset = 0) {
    const res = await apiClient.get('/api/videos', {
      params: { limit, offset },
    });
    return res.data;
  },

  // 6. Video Events
  async getVideoEvents(videoId) {
    const res = await apiClient.get(`/api/videos/${videoId}/events`);
    return res.data;
  },

  // 7. Cross-Video Filtered Events
  async getEvents({ activity, videoId, minConfidence, limit = 50, offset = 0 } = {}) {
    const params = { limit, offset };
    if (activity && activity !== 'ALL') params.activity = activity;
    if (videoId && videoId.trim() !== '') params.video_id = videoId.trim();
    if (minConfidence !== undefined && minConfidence !== null && minConfidence > 0) {
      params.min_confidence = minConfidence;
    }

    const res = await apiClient.get('/api/events', { params });
    return res.data;
  },

  // 8. Semantic Search & Grounded AI Chatbot
  async chatQuery({ query, video_id, activity, min_confidence, limit = 10 } = {}) {
    const res = await apiClient.post('/api/chat/query', {
      query,
      video_id: video_id || null,
      activity: activity && activity !== 'ALL' ? activity : null,
      min_confidence: min_confidence || null,
      limit,
    });
    return res.data;
  },
};

export default api;
