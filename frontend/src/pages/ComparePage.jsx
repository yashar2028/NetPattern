import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { errorMessage } from "../api/client";
import { getRun } from "../api/runsApi";
import { getVersionById } from "../api/sandboxesApi";
import LineChart from "../components/LineChart";
import StatusBadge from "../components/StatusBadge";
import { FINISHED_STATUSES } from "../constants";
import { formatMetric } from "../utils/format";

// The validated categorical palette has eight slots; more runs are compared in the tables only.
const MAX_CHART_RUNS = 8;
const MAX_CELL = 60;

/** Pipeline settings as dotted paths: "trainer.optimizer.lr" → 0.001. */
function flatten(value, prefix, out) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [key, inner] of Object.entries(value)) flatten(inner, prefix ? `${prefix}.${key}` : key, out);
  } else {
    out[prefix] = JSON.stringify(value);
  }
  return out;
}

function settings(spec) {
  if (!spec) return {};
  const out = flatten(spec.task, "task", {});
  for (const node of spec.nodes || []) flatten(node.params, node.type, out);
  return out;
}

function label(run, index) {
  const values = Object.entries(run.labels || {})
    .map(([key, value]) => `${key.split(".").pop()}=${JSON.stringify(value)}`)
    .join(" ");
  return `#${index + 1} ${values || `${run.kind} ${run.id.slice(4, 10)}`}`;
}

export default function ComparePage() {
  const [params] = useSearchParams();
  const ids = useMemo(() => (params.get("runs") || "").split(",").filter(Boolean), [params]);
  const [runs, setRuns] = useState([]);
  const [specs, setSpecs] = useState({});
  const [metric, setMetric] = useState("");
  const [error, setError] = useState("");

  const [tick, setTick] = useState(0);
  const unfinished = runs.some((run) => !FINISHED_STATUSES.includes(run.status));

  useEffect(() => {
    Promise.all(ids.map((id) => getRun(id)))
      .then(async (loaded) => {
        setRuns(loaded);
        const versionIds = [...new Set(loaded.map((run) => run.pipeline_version_id).filter(Boolean))];
        const versions = await Promise.all(versionIds.map((id) => getVersionById(id).catch(() => null)));
        setSpecs(Object.fromEntries(versions.filter(Boolean).map((v) => [v.id, v.spec])));
      })
      .catch((caught) => setError(errorMessage(caught)));
  }, [ids, tick]);

  // Sweeps start many runs at once: refresh until all have finished.
  useEffect(() => {
    if (!unfinished) return undefined;
    const timer = window.setTimeout(() => setTick((value) => value + 1), 5000);
    return () => window.clearTimeout(timer);
  }, [unfinished, tick]);

  const metrics = useMemo(
    () => [
      ...new Set(runs.flatMap((run) => (run.result?.training?.history || []).flatMap((e) => Object.keys(e.metrics)))),
    ],
    [runs]
  );
  const chosen = metric || metrics.find((name) => name.startsWith("val/") && name !== "val/loss") || metrics[0];
  const series = useMemo(
    () =>
      runs.slice(0, MAX_CHART_RUNS).map((run, index) => ({
        name: label(run, index),
        data: (run.result?.training?.history || []).map((epoch) => [epoch.epoch, epoch.metrics[chosen]]),
      })),
    [runs, chosen]
  );

  const finals = useMemo(() => {
    const split = (run) => run.result?.evaluation?.test || run.result?.evaluation?.val;
    const names = [...new Set(runs.flatMap((run) => Object.keys(split(run)?.metrics || {})))];
    return names.map((name) => {
      const values = runs.map((run) => split(run)?.metrics?.[name]);
      const numbers = values.filter((value) => typeof value === "number");
      const mean = numbers.reduce((total, value) => total + value, 0) / (numbers.length || 1);
      const spread = Math.sqrt(numbers.reduce((total, value) => total + (value - mean) ** 2, 0) / (numbers.length || 1));
      return { name, values, mean: numbers.length > 1 ? mean : null, std: spread };
    });
  }, [runs]);

  const differences = useMemo(() => {
    const rows = runs.map((run) => ({
      ...settings(specs[run.pipeline_version_id]),
      // A sweep changes values on top of the saved version.
      ...Object.fromEntries(Object.entries(run.labels || {}).map(([key, value]) => [key, JSON.stringify(value)])),
      "environment.engine": JSON.stringify(run.result?.environment?.engine ?? null),
      "environment.torch": JSON.stringify(run.result?.environment?.torch ?? null),
      "environment.hash": JSON.stringify(run.result?.env_hash ?? null),
      "environment.hardware": JSON.stringify(run.hardware_tier),
    }));
    const keys = [...new Set(rows.flatMap((row) => Object.keys(row)))].sort();
    return keys
      .filter((key) => new Set(rows.map((row) => row[key] ?? "—")).size > 1)
      .map((key) => ({ key, values: rows.map((row) => row[key] ?? "—") }));
  }, [runs, specs]);

  if (error) return <p className="error page-message">{error}</p>;
  if (ids.length < 2) return <p className="muted page-message">Pick at least two runs to compare on a sandbox page.</p>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Compare runs</h1>
          <p className="muted">{runs.length} runs · differences in settings and environment are listed below.</p>
        </div>
      </header>

      <div className="table-scroll">
        <table className="table compact">
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Best epoch</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run, index) => (
              <tr key={run.id}>
                <td>
                  <span className="swatch" style={{ background: index < MAX_CHART_RUNS ? `var(--series-${index + 1})` : "transparent" }} />
                  <Link to={`/runs/${run.id}`}>{label(run, index)}</Link>
                </td>
                <td>
                  <StatusBadge status={run.status} />
                </td>
                <td>{run.result?.training?.best_epoch ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {metrics.length > 0 && (
        <figure className="panel chart-panel">
          <figcaption>
            <select aria-label="Metric" value={chosen} onChange={(event) => setMetric(event.target.value)}>
              {metrics.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>{" "}
            per epoch{runs.length > MAX_CHART_RUNS ? ` (first ${MAX_CHART_RUNS} runs)` : ""}
          </figcaption>
          <LineChart series={series} xName="epoch" label={`${chosen} per epoch for each run`} />
        </figure>
      )}

      {finals.length > 0 && (
        <section className="section">
          <h2>Final evaluation (test, or val)</h2>
          <div className="table-scroll">
            <table className="table compact">
              <thead>
                <tr>
                  <th>metric</th>
                  {runs.map((run, index) => (
                    <th key={run.id} className="number">
                      {label(run, index)}
                    </th>
                  ))}
                  {runs.length > 1 && <th className="number">mean ± std</th>}
                </tr>
              </thead>
              <tbody>
                {finals.map((row) => (
                  <tr key={row.name}>
                    <th scope="row">{row.name}</th>
                    {row.values.map((value, index) => (
                      <td key={index} className="number">
                        {formatMetric(value)}
                      </td>
                    ))}
                    {runs.length > 1 && (
                      <td className="number">
                        {row.mean === null ? "—" : `${formatMetric(row.mean)} ± ${formatMetric(row.std)}`}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="section">
        <h2>What differs</h2>
        {differences.length === 0 ? (
          <p className="muted">Same settings and environment.</p>
        ) : (
          <div className="table-scroll">
            <table className="table compact">
              <thead>
                <tr>
                  <th>setting</th>
                  {runs.map((run, index) => (
                    <th key={run.id}>{label(run, index)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {differences.map((row) => (
                  <tr key={row.key}>
                    <th scope="row">
                      <code>{row.key}</code>
                    </th>
                    {row.values.map((value, index) => (
                      <td key={index} className="small" title={value.length > MAX_CELL ? value : undefined}>
                        {value.length > MAX_CELL ? `${value.slice(0, MAX_CELL)}…` : value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
