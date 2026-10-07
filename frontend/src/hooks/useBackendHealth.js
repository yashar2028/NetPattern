import { useEffect, useState } from "react";

import { getHealth } from "../api/systemApi";
import { HEALTH_POLL_MS } from "../constants";

/**
 * Polls GET /health.
 * status: "checking" | "ok" | "degraded" (API up, database down) | "unreachable"
 */
export function useBackendHealth() {
  const [health, setHealth] = useState({ status: "checking", version: null });

  useEffect(() => {
    let mounted = true;

    async function loadHealth() {
      try {
        const data = await getHealth();
        if (mounted) {
          setHealth({ status: data.status || "ok", version: data.version || null });
        }
      } catch (error) {
        if (!mounted) return;
        const apiAnswered = error.response?.status === 503;
        setHealth({ status: apiAnswered ? "degraded" : "unreachable", version: null });
      }
    }

    loadHealth();
    const timer = window.setInterval(loadHealth, HEALTH_POLL_MS);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  return health;
}
