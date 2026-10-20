import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import OverviewPage from './pages/OverviewPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';
import EventsPage from './pages/EventsPage';
import SystemStatusPage from './pages/SystemStatusPage';
import ChatPage from './pages/ChatPage';

export function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/analysis" element={<VideoAnalysisPage />} />
          <Route path="/events" element={<EventsPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/status" element={<SystemStatusPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
