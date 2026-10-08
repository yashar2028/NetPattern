import { CheckCircle2, CircleDashed, Loader2, Slash, XCircle } from "lucide-react";

const ICONS = {
  queued: CircleDashed,
  running: Loader2,
  completed: CheckCircle2,
  failed: XCircle,
  cancelled: Slash,
};

/** Run status with an icon and a word, never color alone. */
export default function StatusBadge({ status }) {
  const Icon = ICONS[status] || CircleDashed;
  return (
    <span className={`badge badge-${status}`}>
      <Icon size={14} aria-hidden="true" className={status === "running" ? "spin" : undefined} />
      {status}
    </span>
  );
}
