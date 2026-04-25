import type { ClassifyResponse } from "../lib/api";

interface TicketCardProps {
  result: ClassifyResponse;
}

const PRIORITY_STYLES: Record<string, string> = {
  P1: "bg-red-100 text-red-700 border border-red-300",
  P2: "bg-amber-100 text-amber-700 border border-amber-300",
  P3: "bg-slate-100 text-slate-700 border border-slate-300",
};

function PriorityBadge({ priority }: { priority: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${PRIORITY_STYLES[priority] ?? PRIORITY_STYLES.P3}`}
    >
      {priority}
    </span>
  );
}

function ConfidenceBar({ intent, confidence }: { intent: string; confidence: number }) {
  const pct = Math.round(confidence * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="w-48 truncate text-xs text-gray-600">{intent}</span>
      <div className="flex-1 rounded-full bg-gray-100 h-2 overflow-hidden">
        <div
          className="h-2 rounded-full bg-blue-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 text-right text-xs font-medium text-gray-700">{pct}%</span>
    </div>
  );
}

export default function TicketCard({ result }: TicketCardProps) {
  const { ticket, top_k } = result;

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Top intent</p>
          <p className="text-base font-semibold text-gray-800">{result.intent}</p>
        </div>
        <PriorityBadge priority={ticket.priority} />
      </div>

      {/* Ticket metadata */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <div>
          <span className="text-gray-400 text-xs">Department</span>
          <p className="font-medium text-gray-700">{ticket.department}</p>
        </div>
        <div>
          <span className="text-gray-400 text-xs">SLA</span>
          <p className="font-medium text-gray-700">{ticket.sla_hours}h</p>
        </div>
        <div className="col-span-2">
          <span className="text-gray-400 text-xs">Ticket ID</span>
          <p className="font-mono text-xs text-gray-500 truncate">{ticket.id}</p>
        </div>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5">
        {ticket.tags.map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Top-k confidence bars */}
      <div className="flex flex-col gap-1.5">
        <p className="text-xs text-gray-400 uppercase tracking-wide">Top 3 predictions</p>
        {top_k.map((score) => (
          <ConfidenceBar
            key={score.intent}
            intent={score.intent}
            confidence={score.confidence}
          />
        ))}
      </div>
    </div>
  );
}
