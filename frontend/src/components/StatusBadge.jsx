import React from 'react';
import { CheckCircle, Clock, XCircle, ShieldAlert } from 'lucide-react';

export const StatusBadge = ({ status }) => {
  const norm = (status || '').toLowerCase();

  switch (norm) {
    case 'completed':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <CheckCircle className="w-3.5 h-3.5" />
          Completed
        </span>
      );
    case 'processing':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 animate-pulse">
          <Clock className="w-3.5 h-3.5 animate-spin" />
          Processing
        </span>
      );
    case 'uploaded':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
          <Clock className="w-3.5 h-3.5" />
          Uploaded
        </span>
      );
    case 'failed':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
          <XCircle className="w-3.5 h-3.5" />
          Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700">
          {status}
        </span>
      );
  }
};

export const ActivityBadge = ({ activity }) => {
  const label = activity || 'Unknown';

  const colorMap = {
    Fighting: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
    Robbery: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    Burglary: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
    Shoplifting: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
    Vandalism: 'bg-fuchsia-500/15 text-fuchsia-400 border-fuchsia-500/30',
    RoadAccidents: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    Normal: 'bg-slate-700/30 text-slate-400 border-slate-600/30',
  };

  const style = colorMap[label] || 'bg-blue-500/15 text-blue-400 border-blue-500/30';

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border ${style}`}>
      <ShieldAlert className="w-3.5 h-3.5" />
      {label}
    </span>
  );
};

export default StatusBadge;
