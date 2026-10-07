import { useBackendHealth } from "../hooks/useBackendHealth";

const LABELS = {
  checking: "backend: checking…",
  ok: "backend: ok",
  degraded: "backend: database down",
  unreachable: "backend: unreachable",
};

export default function BackendStatus() {
  const { status, version } = useBackendHealth();

  return (
    <span
      className={`status-pill status-${status}`}
      role="status"
      title={version ? `API version ${version}` : undefined}
    >
      <span className="status-dot" aria-hidden="true" />
      {LABELS[status] ?? LABELS.unreachable}
    </span>
  );
}
