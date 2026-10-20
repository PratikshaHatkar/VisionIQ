import React from 'react';

export const StatsCard = ({ title, value, subtitle, icon: Icon, badge, accentColor = 'blue' }) => {
  const colorStyles = {
    blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    rose: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    indigo: 'text-indigo-400 bg-indigo-500/10 border-indigo-500/20',
  };

  const currentStyle = colorStyles[accentColor] || colorStyles.blue;

  return (
    <div className="bg-[#111827] border border-slate-800/90 rounded-2xl p-5 shadow-sm hover:border-slate-700/80 transition-all">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {title}
        </span>
        {Icon && (
          <div className={`p-2 rounded-xl border ${currentStyle}`}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-baseline justify-between">
        <div className="text-2xl font-bold text-slate-100 tracking-tight">
          {value}
        </div>
        {badge && (
          <span className="text-xs px-2 py-0.5 rounded-md font-medium bg-slate-800 text-slate-300 border border-slate-700">
            {badge}
          </span>
        )}
      </div>

      {subtitle && (
        <div className="mt-2 text-xs text-slate-400 truncate">
          {subtitle}
        </div>
      )}
    </div>
  );
};

export default StatsCard;
