import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Bot,
  User,
  Sparkles,
  ShieldAlert,
  Eye,
  Filter,
  Info,
  RefreshCw,
  Lightbulb
} from 'lucide-react';
import { api, getEvidenceUrl } from '../api/client';
import { ActivityBadge } from '../components/StatusBadge';
import EvidenceModal from '../components/EvidenceModal';

const SUGGESTIONS = [
  'Show fighting incidents',
  'Find incidents with confidence above 70%',
  'Show all road accidents',
  'Find burglary events',
  'Show incidents between 6 PM and 10 PM',
  'Show all shoplifting incidents'
];

export const ChatPage = () => {
  const [inputQuery, setInputQuery] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Hello, I am VisionIQ Surveillance AI. Ask me about recorded CCTV incidents using natural language. I will retrieve matching events from MongoDB and link every finding to genuine evidence snapshots.',
      events: [],
      applied_filters: null,
      limitations: []
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (queryText) => {
    const q = (queryText || inputQuery).trim();
    if (!q || loading) return;

    setInputQuery('');

    // Append user query message
    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: q
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const data = await api.chatQuery({ query: q, limit: 8 });

      const assistantMsg = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        text: data.answer,
        applied_filters: data.applied_filters,
        events: data.events || [],
        total: data.total || 0,
        retrieval_method: data.retrieval_method,
        limitations: data.limitations || []
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message || 'Failed to query surveillance AI assistant.';
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          text: `Error processing query: ${errMsg}`,
          events: [],
          applied_filters: null,
          limitations: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] animate-in fade-in duration-200">
      {/* Header */}
      <div className="pb-3 border-b border-slate-800 shrink-0">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          Semantic Search & AI Assistant
          <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center gap-1">
            <Sparkles className="w-3 h-3" /> Hybrid Retrieval
          </span>
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 mt-1">
          Ask questions in natural language. Detections are strictly grounded in MongoDB event records with evidence snapshots.
        </p>
      </div>

      {/* Chat Messages Feed */}
      <div className="flex-1 overflow-y-auto py-4 space-y-6 pr-2">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3.5 max-w-4xl ${
              msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
            }`}
          >
            {/* Avatar */}
            <div
              className={`w-8 h-8 rounded-xl shrink-0 flex items-center justify-center border ${
                msg.role === 'user'
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-[#1E293B] border-slate-700 text-blue-400'
              }`}
            >
              {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            {/* Bubble Content */}
            <div
              className={`rounded-2xl p-4 text-xs sm:text-sm space-y-3 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-tr-none'
                  : 'bg-[#111827] border border-slate-800 text-slate-200 rounded-tl-none shadow-sm'
              }`}
            >
              <div className="leading-relaxed">{msg.text}</div>

              {/* Applied Filters Badge Bar */}
              {msg.applied_filters && (
                <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-800/80 text-[11px]">
                  <span className="text-slate-400 font-medium flex items-center gap-1">
                    <Filter className="w-3 h-3 text-blue-400" /> Filters:
                  </span>
                  {msg.applied_filters.activity && (
                    <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30 font-semibold">
                      Activity: {msg.applied_filters.activity}
                    </span>
                  )}
                  {msg.applied_filters.min_confidence && (
                    <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 font-mono">
                      Conf &ge; {(msg.applied_filters.min_confidence * 100).toFixed(0)}%
                    </span>
                  )}
                  {msg.applied_filters.time_filter && (
                    <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      Time: {msg.applied_filters.time_filter}
                    </span>
                  )}
                  {msg.applied_filters.video_id && (
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                      Video: {msg.applied_filters.video_id.slice(0, 8)}...
                    </span>
                  )}
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] ml-auto">
                    Method: {msg.retrieval_method || 'hybrid'}
                  </span>
                </div>
              )}

              {/* Matching Event Cards Grid */}
              {msg.events && msg.events.length > 0 && (
                <div className="pt-2 border-t border-slate-800 space-y-2">
                  <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                    Grounded Evidence Records ({msg.events.length})
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                    {msg.events.map((ev) => (
                      <div
                        key={ev.event_id}
                        className="bg-[#0F172A] border border-slate-800 rounded-xl p-2.5 space-y-2 hover:border-slate-700 transition-all flex flex-col justify-between"
                      >
                        <div className="flex gap-2.5">
                          {/* Evidence Snapshot Thumbnail */}
                          <div
                            onClick={() => setSelectedEvent(ev)}
                            className="w-16 h-12 rounded-lg bg-slate-900 border border-slate-700 overflow-hidden shrink-0 cursor-pointer relative group"
                          >
                            <img
                              src={getEvidenceUrl(ev.evidence_snapshot)}
                              alt={ev.activity_label}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                              onError={(e) => {
                                e.target.style.display = 'none';
                              }}
                            />
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                              <Eye className="w-3 h-3 text-white" />
                            </div>
                          </div>

                          <div className="flex-1 min-w-0 space-y-1">
                            <div className="flex items-center justify-between">
                              <ActivityBadge activity={ev.activity_label} />
                              <span className="text-[11px] font-bold text-slate-200">
                                {(ev.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="text-[11px] text-slate-400 font-mono truncate">
                              {ev.start_time_formatted} &rarr; {ev.end_time_formatted}
                            </div>
                          </div>
                        </div>

                        <div className="pt-1.5 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-500">
                          <span className="truncate max-w-[130px]" title={ev.filename || ev.video_id}>
                            {ev.filename || ev.video_id}
                          </span>
                          <button
                            onClick={() => setSelectedEvent(ev)}
                            className="text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-0.5"
                          >
                            <Eye className="w-2.5 h-2.5" /> Enlarge
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Limitations & Verification Notices */}
              {msg.limitations && msg.limitations.length > 0 && (
                <div className="pt-2 border-t border-slate-800 space-y-1 text-[11px] text-slate-400 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80">
                  {msg.limitations.map((lim, idx) => (
                    <div key={idx} className="flex items-start gap-1.5">
                      <Info className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
                      <span>{lim}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3.5 max-w-4xl">
            <div className="w-8 h-8 rounded-xl bg-[#1E293B] border border-slate-700 text-blue-400 flex items-center justify-center shrink-0">
              <Bot className="w-4 h-4 animate-pulse" />
            </div>
            <div className="bg-[#111827] border border-slate-800 rounded-2xl rounded-tl-none p-4 text-xs text-slate-400 flex items-center gap-2">
              <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
              <span>Analyzing query, filtering MongoDB records, and ranking relevance...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Suggestion Pills */}
      <div className="py-2.5 shrink-0 border-t border-slate-800 flex items-center gap-2 overflow-x-auto">
        <span className="text-[11px] text-slate-500 font-medium flex items-center gap-1 shrink-0">
          <Lightbulb className="w-3 h-3 text-amber-400" /> Suggestions:
        </span>
        {SUGGESTIONS.map((sug, i) => (
          <button
            key={i}
            onClick={() => handleSend(sug)}
            disabled={loading}
            className="px-2.5 py-1 text-xs rounded-full bg-[#111827] hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 transition-colors shrink-0 disabled:opacity-40"
          >
            {sug}
          </button>
        ))}
      </div>

      {/* Query Input Box */}
      <div className="pt-2 shrink-0">
        <div className="relative flex items-center">
          <textarea
            rows={1}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about suspicious CCTV activities (e.g. 'Show fighting incidents after 8 PM')..."
            className="w-full bg-[#111827] border border-slate-800 focus:border-blue-500 rounded-xl px-4 py-3 pr-14 text-xs sm:text-sm text-white placeholder-slate-500 focus:outline-none resize-none shadow-inner"
          />
          <button
            onClick={() => handleSend()}
            disabled={!inputQuery.trim() || loading}
            className="absolute right-2.5 p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-all disabled:opacity-40"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Evidence Snapshot Modal */}
      {selectedEvent && (
        <EvidenceModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </div>
  );
};

export default ChatPage;
