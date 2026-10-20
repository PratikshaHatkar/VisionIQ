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

// Helper to get live camera MJPEG stream URL
export const getLiveStreamUrl = (cameraId) => {
  if (!cameraId) return null;
  return `${BASE_URL}/api/live/cameras/${cameraId}/stream`;
};

// Helper to get WebSocket URL for live notifications
export const getWebSocketUrl = (ownerId = 'default_owner') => {
  const wsProtocol = BASE_URL.startsWith('https') ? 'wss' : 'ws';
  const cleanHost = BASE_URL.replace(/^https?:\/\//, '');
  return `${wsProtocol}://${cleanHost}/api/live/ws?owner_id=${encodeURIComponent(ownerId)}`;
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

  // 9. Live CCTV Surveillance APIs
  async listCameras(ownerId) {
    const params = ownerId ? { owner_id: ownerId } : {};
    const res = await apiClient.get('/api/live/cameras', { params });
    return res.data;
  },

  async registerCamera(cameraData) {
    const res = await apiClient.post('/api/live/cameras', cameraData);
    return res.data;
  },

  async updateCamera(cameraId, updates) {
    const res = await apiClient.patch(`/api/live/cameras/${cameraId}`, updates);
    return res.data;
  },

  async deleteCamera(cameraId) {
    const res = await apiClient.delete(`/api/live/cameras/${cameraId}`);
    return res.data;
  },

  async startCamera(cameraId) {
    const res = await apiClient.post(`/api/live/cameras/${cameraId}/start`);
    return res.data;
  },

  async stopCamera(cameraId) {
    const res = await apiClient.post(`/api/live/cameras/${cameraId}/stop`);
    return res.data;
  },

  async listLiveIncidents({ ownerId, cameraId, activityLabel, limit = 50 } = {}) {
    const params = { limit };
    if (ownerId) params.owner_id = ownerId;
    if (cameraId) params.camera_id = cameraId;
    if (activityLabel && activityLabel !== 'ALL') params.activity_label = activityLabel;
    const res = await apiClient.get('/api/live/incidents', { params });
    return res.data;
  },

  async acknowledgeLiveIncident(incidentId, ownerId) {
    const params = ownerId ? { owner_id: ownerId } : {};
    const res = await apiClient.patch(`/api/live/incidents/${incidentId}/acknowledge`, null, { params });
    return res.data;
  },

  async listLiveNotifications({ ownerId, isRead, limit = 50 } = {}) {
    const params = { limit };
    if (ownerId) params.owner_id = ownerId;
    if (isRead !== undefined && isRead !== null) params.is_read = isRead;
    const res = await apiClient.get('/api/live/notifications', { params });
    return res.data;
  },

  async markNotificationRead(notificationId, ownerId) {
    const params = ownerId ? { owner_id: ownerId } : {};
    const res = await apiClient.patch(`/api/live/notifications/${notificationId}/read`, null, { params });
    return res.data;
  },

  async markAllNotificationsRead(ownerId) {
    const res = await apiClient.patch('/api/live/notifications/read-all', null, {
      params: { owner_id: ownerId },
    });
    return res.data;
  },
};

export default api;
