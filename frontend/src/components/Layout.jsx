import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Video,
  ShieldAlert,
  Activity,
  MessageSquare,
  Menu,
  X,
  Radio,
  ExternalLink,
} from 'lucide-react';
import { api } from '../api/client';

export const Layout = ({ children }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [systemHealth, setSystemHealth] = useState({ status: 'checking', engine: null });

  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const data = await api.getHealth();
        if (isMounted) {
          setSystemHealth({
            status: 'online',
            engine: data.database_engine || 'connected',
            device: data.model?.device || 'cpu',
          });
        }
      } catch {
        if (isMounted) {
          setSystemHealth({ status: 'offline', engine: 'unreachable' });
        }
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000); // Check every 15s
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { to: '/', label: 'Overview', icon: LayoutDashboard },
    { to: '/analysis', label: 'Video Analysis', icon: Video },
    { to: '/events', label: 'Detected Events', icon: ShieldAlert },
    { to: '/chat', label: 'AI Assistant', icon: MessageSquare },
    { to: '/status', label: 'System Status', icon: Activity },
  ];

  return (
    <div className="min-h-screen bg-[#0B0F17] flex text-slate-200">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex flex-col w-64 bg-[#0F172A] border-r border-slate-800/80 fixed inset-y-0 z-40">
        {/* Brand */}
        <div className="h-16 flex items-center px-6 border-b border-slate-800/80 gap-3">
          <div className="p-2 rounded-xl bg-blue-600/20 border border-blue-500/30 text-blue-400">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-base tracking-tight text-white">VisionIQ</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 font-semibold border border-blue-500/30">
                PRO
              </span>
            </div>
            <div className="text-[11px] text-slate-400">AI Surveillance System</div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Navigation
          </div>
          {navLinks.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
                  }`
                }
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Bottom System Status Widget */}
        <div className="p-4 border-t border-slate-800/80">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    systemHealth.status === 'online'
                      ? 'bg-emerald-400 ring-4 ring-emerald-400/20'
                      : systemHealth.status === 'checking'
                      ? 'bg-amber-400 animate-ping'
                      : 'bg-rose-500 ring-4 ring-rose-500/20'
                  }`}
                />
                <span className="text-xs font-semibold text-slate-300">
                  {systemHealth.status === 'online' ? 'Backend Live' : systemHealth.status === 'checking' ? 'Connecting...' : 'Backend Offline'}
                </span>
              </div>
              <span className="text-[10px] font-mono text-slate-400 uppercase">
                {systemHealth.engine || 'v1.0.0'}
              </span>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between">
              <span>Host: 127.0.0.1:8000</span>
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="text-blue-400 hover:text-blue-300 flex items-center gap-0.5"
                title="Open OpenAPI Swagger Docs"
              >
                API <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 md:pl-64 flex flex-col min-w-0">
        {/* Mobile Header */}
        <header className="md:hidden h-16 bg-[#0F172A] border-b border-slate-800 px-4 flex items-center justify-between sticky top-0 z-30">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-blue-600/20 text-blue-400">
              <Radio className="w-5 h-5" />
            </div>
            <span className="font-bold text-white tracking-tight">VisionIQ</span>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
            aria-label="Toggle Navigation"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </header>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-[#0F172A] border-b border-slate-800 px-4 py-4 space-y-1">
            {navLinks.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  onClick={() => setMobileMenuOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        )}

        {/* Page Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
