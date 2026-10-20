import React from 'react';
import { Inbox } from 'lucide-react';

export const EmptyState = ({
  icon: Icon = Inbox,
  title = 'No records found',
  description = 'There are no items to display at this time.',
  action,
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center bg-[#111827]/40 border border-dashed border-slate-800 rounded-2xl">
      <div className="p-3.5 bg-slate-800/60 rounded-2xl text-slate-400 mb-3 border border-slate-700/50">
        <Icon className="w-6 h-6" />
      </div>
      <h4 className="text-sm font-semibold text-slate-200 mb-1">{title}</h4>
      <p className="text-xs text-slate-400 max-w-sm mb-4">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
};

export default EmptyState;
