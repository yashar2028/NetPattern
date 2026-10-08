export function formatDate(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function formatBytes(bytes) {
  if (bytes == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatSeconds(seconds) {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

export function formatMetric(value) {
  if (value == null) return "—";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4);
}

export function formatCount(value) {
  if (value == null) return "—";
  return new Intl.NumberFormat().format(value);
}
