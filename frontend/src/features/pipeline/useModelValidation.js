import { useEffect, useMemo, useState } from "react";

import { errorIssues, errorMessage } from "../../api/client";
import { sessionCall } from "../../api/sandboxesApi";

export default function useModelValidation(sandboxId, architecture, taskType, outputs, enabled) {
  const [response, setResponse] = useState(null);
  const [attempt, setAttempt] = useState(0);
  const request = useMemo(() => enabled ? {
    sandboxId, attempt, params: { architecture, task: { type: taskType }, num_outputs: outputs },
  } : null, [sandboxId, architecture, taskType, outputs, enabled, attempt]);

  useEffect(() => {
    if (!request) return undefined;
    let active = true;
    const timer = window.setTimeout(async () => {
      try {
        const analysis = await sessionCall(request.sandboxId, "validate_architecture", request.params);
        if (active) setResponse({ request, analysis });
      } catch (caught) {
        const issues = errorIssues(caught);
        if (active) setResponse({
          request,
          analysis: issues.length ? { ok: false, issues } : null,
          error: issues.length ? null : errorMessage(caught),
        });
      }
    }, 350);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [request]);

  const current = request && response?.request === request ? response : null;
  return {
    analysis: current?.analysis,
    error: current?.error,
    checking: Boolean(request && !current),
    retry: () => setAttempt((value) => value + 1),
  };
}
