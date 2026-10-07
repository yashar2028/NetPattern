const runtimeApiUrl = (() => {
  if (import.meta.env.VITE_API_BASE_URL) return import.meta.env.VITE_API_BASE_URL;
  if (typeof window === "undefined") return "http://localhost:8000";
  const { hostname, protocol } = window.location;
  // Codespaces: swap -5173 for -8000 in the hostname
  if (hostname.includes("-5173.")) {
    return `${protocol}//${hostname.replace("-5173.", "-8000.")}`;
  }
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }
  return `${protocol}//${hostname}:8000`;
})();

export const API_BASE_URL = runtimeApiUrl;

export const HEALTH_POLL_MS = 15000;
