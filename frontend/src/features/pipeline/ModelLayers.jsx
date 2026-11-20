import { Fragment, useEffect, useState } from "react";
import { Loader2, Lock, X } from "lucide-react";

import { errorMessage } from "../../api/client";
import { sessionCall } from "../../api/sandboxesApi";
import { formatCount } from "../../utils/format";

/** Read-only list of a zoo model's layers, as the engine builds it for this task. */
export default function ModelLayers({ sandboxId, task, architecture, onClose }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    // The real head size comes from the dataset; the preview uses the listed classes or 2.
    sessionCall(sandboxId, "inspect_model", { architecture, task, num_outputs: task?.classes?.length || 2 })
      .then((value) => active && setResult(value))
      .catch((caught) => active && setError(errorMessage(caught)));
    return () => {
      active = false;
    };
  }, [sandboxId, task, architecture]);

  const trainable = Object.fromEntries((result?.modules || []).map((m) => [m.name, m.trainable]));
  const rows = result?.traceable
    ? result.nodes.filter((node) => node.kind === "call_module" || node.kind === "call_function")
    : (result?.modules || []).filter((m) => m.leaf).map((m) => ({ ...m, target: m.name, kind: "call_module" }));
  const total = (result?.modules || []).filter((m) => !m.name.includes(".")).reduce((sum, m) => sum + m.params, 0);
  const modules = rows.filter((row) => row.kind === "call_module");
  const head = modules[modules.length - 1];
  // A heading where a new top-level block (layer1, layer2, …) starts.
  const headings = new Set();
  modules.reduce((previous, row) => {
    const block = row.target.includes(".") ? row.target.split(".")[0] : null;
    if (block && block !== previous) headings.add(row);
    return block;
  }, null);

  return (
    <div className="layer-editor model-layers" role="dialog" aria-label="Model layers">
      <header className="layer-toolbar">
        <h2>Layers of {architecture.base?.name}</h2>
        <span className="badge">
          <Lock size={12} aria-hidden="true" /> read-only
        </span>
        <div className="layer-summary">
          {result && `${modules.length} layers · ${formatCount(total)} params`}
        </div>
        <button type="button" className="button primary" onClick={onClose}>
          <X size={16} aria-hidden="true" /> Close
        </button>
      </header>
      <p className="muted small">
        A ready-made model&apos;s layers can&apos;t be changed. NetPattern only replaces the last layer so it predicts your
        classes. In the Model panel you can pick another model, start from blank weights, or freeze the pretrained part
        so only the new last layer learns. To design layers yourself, use &quot;Build custom layers instead&quot;.
      </p>
      {error && <p className="error">{error}</p>}
      {!result && !error && (
        <p className="muted">
          <Loader2 size={16} className="spin inline-icon" aria-hidden="true" /> Reading the model…
        </p>
      )}
      {result && (
        <div className="table-scroll">
          <table className="table compact">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Type</th>
                <th>Output (C × H × W)</th>
                <th className="number">Parameters</th>
                <th>Learns in training</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                  <Fragment key={row.id || row.name}>
                    {headings.has(row) && (
                      <tr className="group-row">
                        <th colSpan={5}>{row.target.split(".")[0]}</th>
                      </tr>
                    )}
                    {row.kind === "call_function" ? (
                      <tr className="function-row">
                        <td colSpan={2}>{row.target === "add" ? "+ skip connection (adds the block's input back)" : row.target}</td>
                        <td>{row.shape?.join(" × ") || "—"}</td>
                        <td />
                        <td />
                      </tr>
                    ) : (
                      <tr className={row === head ? "head-row" : undefined}>
                        <td>
                          {row.target}
                          {row === head && " · replaced to predict your classes"}
                        </td>
                        <td>{row.type}</td>
                        <td>{row.shape?.join(" × ") || "—"}</td>
                        <td className="number">{row.params ? formatCount(row.params) : "—"}</td>
                        <td>{!row.params ? "—" : trainable[row.target] === false ? "frozen" : "yes"}</td>
                      </tr>
                    )}
                  </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
