import { useState } from "react";
import { Plus, Trash2, X } from "lucide-react";

const PRESETS = [
  { label: "Learning rate", path: "trainer.optimizer.lr", values: "0.01, 0.001, 0.0001" },
  { label: "Batch size", path: "trainer.batch_size", values: "8, 16, 32" },
  { label: "Epochs", path: "trainer.epochs", values: "5, 10, 20" },
  { label: "Weight decay", path: "trainer.optimizer.weight_decay", values: "0, 0.01, 0.1" },
];

/** "0.01, 1e-3, true, adam" → [0.01, 0.001, true, "adam"]. */
function parseValues(text) {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      try {
        return JSON.parse(part);
      } catch {
        return part;
      }
    });
}

/** Many runs of the saved pipeline with chosen values changed; the results open on Compare. */
export default function SweepPanel({ tiers, onStart, onClose }) {
  const [rows, setRows] = useState([{ ...PRESETS[0] }]);
  const [folds, setFolds] = useState("");
  const [mode, setMode] = useState("grid");
  const [maxRuns, setMaxRuns] = useState(16);
  const [tier, setTier] = useState(tiers[0]?.name || "cpu");
  const [busy, setBusy] = useState(false);

  const parameters = [
    ...rows.filter((row) => row.path && row.values).map((row) => ({ path: row.path, values: parseValues(row.values) })),
    ...(Number(folds) >= 2
      ? [
          { path: "dataset.split.folds", values: [Number(folds)] },
          { path: "dataset.split.fold", values: [...Array(Number(folds)).keys()] },
        ]
      : []),
  ];
  const runs = parameters.reduce((total, p) => total * p.values.length, parameters.length ? 1 : 0);
  const setRow = (index, changes) => setRows(rows.map((row, i) => (i === index ? { ...row, ...changes } : row)));

  return (
    <section className="panel sweep-panel" aria-label="Sweep">
      <div className="list-item-header">
        <h2>Sweep</h2>
        <button type="button" className="icon-button" aria-label="close" onClick={onClose}>
          <X size={16} aria-hidden="true" />
        </button>
      </div>
      <p className="muted small">
        Runs the saved version once per combination. A path is a node and a field, e.g. <code>trainer.optimizer.lr</code>.
      </p>
      {rows.map((row, index) => (
        <div key={index} className="inline-form compact">
          <input aria-label="path" value={row.path} placeholder="trainer.optimizer.lr" onChange={(e) => setRow(index, { path: e.target.value })} />
          <input aria-label="values" className="grow-input" value={row.values} placeholder="comma-separated values" onChange={(e) => setRow(index, { values: e.target.value })} />
          <button type="button" className="icon-button" aria-label="remove" onClick={() => setRows(rows.filter((_, i) => i !== index))}>
            <Trash2 size={14} aria-hidden="true" />
          </button>
        </div>
      ))}
      <div className="toolbar">
        {PRESETS.map((preset) => (
          <button key={preset.path} type="button" className="button ghost" onClick={() => setRows([...rows, { ...preset }])}>
            <Plus size={14} aria-hidden="true" /> {preset.label}
          </button>
        ))}
      </div>
      <div className="inline-form">
        <label className="field">
          <span>Cross-validation folds (optional)</span>
          <input type="number" min="2" max="20" value={folds} placeholder="e.g. 5" onChange={(e) => setFolds(e.target.value)} />
        </label>
        <label className="field">
          <span>Combinations</span>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="grid">all (grid)</option>
            <option value="random">random sample</option>
          </select>
        </label>
        <label className="field">
          <span>At most</span>
          <input type="number" min="1" max="32" value={maxRuns} onChange={(e) => setMaxRuns(Number(e.target.value))} />
        </label>
        <label className="field">
          <span>Hardware</span>
          <select value={tier} onChange={(e) => setTier(e.target.value)}>
            {tiers.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="button primary"
          disabled={busy || runs === 0}
          onClick={async () => {
            setBusy(true);
            try {
              await onStart({ parameters, mode, max_runs: maxRuns, hardware_tier: tier });
            } finally {
              setBusy(false);
            }
          }}
        >
          Start {mode === "grid" ? runs : Math.min(runs, maxRuns)} runs
        </button>
      </div>
    </section>
  );
}
