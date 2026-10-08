import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { CircleStop, RotateCcw } from "lucide-react";

import { errorMessage } from "../api/client";
import { cancelRun, getRun, rerun, streamRunEvents } from "../api/runsApi";
import LineChart from "../components/LineChart";
import StatusBadge from "../components/StatusBadge";
import { FINISHED_STATUSES } from "../constants";
import { formatCount, formatDate, formatMetric, formatSeconds } from "../utils/format";

/** Everything the page shows while a run is live, rebuilt from its event log. */
function summarize(events) {
  const view = { phases: [], history: [], evaluation: {}, warnings: [], sanity: [] };
  for (const { type, data } of events) {
    switch (type) {
      case "phase":
        view.phases.push(data.name);
        break;
      case "environment_ready":
        view.environment = data;
        break;
      case "dataset_indexed":
        view.dataset = data;
        break;
      case "model_built":
        view.model = data;
        break;
      case "batch":
        view.batch = data;
        break;
      case "epoch_end":
        view.history.push(data);
        break;
      case "evaluation":
        view.evaluation[data.split] = { metrics: data.metrics };
        break;
      case "sanity_step":
        view.sanity.push([data.step, data.loss]);
        break;
      case "warning":
        view.warnings.push(data.message);
        break;
      case "run_failed":
        view.failure = data;
        break;
      default:
    }
  }
  return view;
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function MetricTable({ metrics }) {
  return (
    <table className="table compact">
      <tbody>
        {Object.entries(metrics).map(([name, value]) => (
          <tr key={name}>
            <th scope="row">{name}</th>
            <td className="number">{formatMetric(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ConfusionMatrix({ labels, matrix }) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="table-scroll">
      <table className="table confusion">
        <caption className="muted small">rows: true class · columns: predicted class</caption>
        <thead>
          <tr>
            <th />
            {labels.map((label) => (
              <th key={label} scope="col">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th scope="row">{labels[i]}</th>
              {row.map((count, j) => (
                <td
                  key={j}
                  className={`number${i === j ? " diagonal" : ""}`}
                  style={{ "--strength": count / max }}
                >
                  {count}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerClassTable({ rows }) {
  const columns = Object.keys(rows[0] || {}).filter((key) => key !== "class");
  return (
    <div className="table-scroll">
      <table className="table compact">
        <thead>
          <tr>
            <th scope="col">class</th>
            {columns.map((column) => (
              <th key={column} scope="col" className="number">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.class}>
              <th scope="row">{row.class}</th>
              {columns.map((column) => (
                <td key={column} className="number">
                  {formatMetric(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrainingView({ history, monitor, bestEpoch }) {
  const valMetrics = useMemo(
    () => [...new Set(history.flatMap((epoch) => Object.keys(epoch.metrics)))].filter((key) => key.startsWith("val/") && key !== "val/loss"),
    [history]
  );
  const [chosen, setChosen] = useState(null);
  const metric = chosen && valMetrics.includes(chosen) ? chosen : valMetrics.includes(monitor) ? monitor : valMetrics[0];

  const lossSeries = useMemo(() => {
    const series = [{ name: "train loss", data: history.map((e) => [e.epoch, e.metrics["train/loss"]]) }];
    if (history.some((e) => e.metrics["val/loss"] != null)) {
      series.push({ name: "validation loss", data: history.map((e) => [e.epoch, e.metrics["val/loss"]]) });
    }
    return series;
  }, [history]);
  const metricSeries = useMemo(
    () => (metric ? [{ name: metric, data: history.map((e) => [e.epoch, e.metrics[metric]]) }] : []),
    [history, metric]
  );
  const columns = [...new Set(history.flatMap((epoch) => Object.keys(epoch.metrics)))];

  return (
    <section className="section">
      <h2>Training</h2>
      <div className="chart-grid">
        <figure className="panel chart-panel">
          <figcaption>Loss per epoch</figcaption>
          <LineChart series={lossSeries} xName="epoch" label="Training and validation loss per epoch" />
        </figure>
        {metric && (
          <figure className="panel chart-panel">
            <figcaption>
              <select aria-label="Metric" value={metric} onChange={(event) => setChosen(event.target.value)}>
                {valMetrics.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>{" "}
              per epoch
            </figcaption>
            <LineChart series={metricSeries} xName="epoch" label={`${metric} per epoch`} />
          </figure>
        )}
      </div>
      <div className="table-scroll">
        <table className="table compact">
          <thead>
            <tr>
              <th scope="col">epoch</th>
              {columns.map((column) => (
                <th key={column} scope="col" className="number">
                  {column}
                </th>
              ))}
              <th scope="col" className="number">
                learning rate
              </th>
              <th scope="col" className="number">
                time
              </th>
            </tr>
          </thead>
          <tbody>
            {history.map((epoch) => (
              <tr key={epoch.epoch}>
                <th scope="row">
                  {epoch.epoch}
                  {epoch.epoch === bestEpoch && <span className="tag">best</span>}
                </th>
                {columns.map((column) => (
                  <td key={column} className="number">
                    {formatMetric(epoch.metrics[column])}
                  </td>
                ))}
                <td className="number">{epoch.lr?.toExponential(2)}</td>
                <td className="number">{formatSeconds(epoch.duration_s)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EvaluationView({ evaluation }) {
  return (
    <section className="section">
      <h2>Evaluation of the best checkpoint</h2>
      {Object.entries(evaluation).map(([split, { metrics, details }]) => (
        <div key={split} className="evaluation">
          <h3>{split}</h3>
          <MetricTable metrics={metrics} />
          {details?.confusion_matrix && <ConfusionMatrix {...details.confusion_matrix} />}
          {details?.per_class?.length > 0 && <PerClassTable rows={details.per_class} />}
        </div>
      ))}
    </section>
  );
}

function SanityView({ result, steps }) {
  const losses = result?.losses ? result.losses.map((loss, index) => [index + 1, loss]) : steps;
  return (
    <section className="section">
      <h2>Sanity check: overfit one batch</h2>
      {result && (
        <p className={result.passed ? "" : "error"}>
          {result.passed ? "Passed" : "Failed"}: loss went from {formatMetric(result.initial_loss)} to{" "}
          {formatMetric(result.final_loss)} in {result.steps} steps.
        </p>
      )}
      {result?.advice && <p className="muted">Advice: {result.advice}.</p>}
      {losses.length > 0 && (
        <figure className="panel chart-panel">
          <figcaption>Loss per step</figcaption>
          <LineChart series={[{ name: "loss", data: losses }]} xName="step" label="Loss per step on one batch" />
        </figure>
      )}
    </section>
  );
}

function ProfileView({ profile }) {
  return (
    <section className="section">
      <h2>Dataset profile</h2>
      <div className="stat-row">
        <Stat label="samples" value={formatCount(profile.samples)} />
        <Stat label="groups" value={formatCount(profile.groups)} />
        {Object.entries(profile.splits || {}).map(([name, count]) => (
          <Stat key={name} label={`${name} split`} value={formatCount(count)} />
        ))}
        {profile.image_size && (
          <Stat
            label="median size (h × w)"
            value={`${profile.image_size.height.median} × ${profile.image_size.width.median}`}
          />
        )}
      </div>
      {profile.warnings?.length > 0 && (
        <ul className="warning-list">
          {profile.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      {profile.class_counts && (
        <PerClassTable rows={Object.entries(profile.class_counts).map(([name, count]) => ({ class: name, samples: count }))} />
      )}
      <details>
        <summary>All profile values</summary>
        <pre className="code-block">{JSON.stringify(profile, null, 2)}</pre>
      </details>
    </section>
  );
}

// A new run id (e.g. after "Run again") mounts a fresh view with empty state.
export default function RunPage() {
  const { runId } = useParams();
  return <RunView key={runId} runId={runId} />;
}

function RunView({ runId }) {
  const navigate = useNavigate();
  const [run, setRun] = useState(null);
  const [events, setEvents] = useState([]);
  const [streamError, setStreamError] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => getRun(runId).then(setRun).catch((caught) => setError(errorMessage(caught)));
    refresh();
    streamRunEvents(runId, {
      signal: controller.signal,
      onEvent: (event) => {
        // Phase changes and the end of the stream also change the run's status and result.
        if (event.type === "end" || event.type === "phase") refresh();
        if (event.type !== "end") setEvents((current) => [...current, event]);
      },
    }).catch((caught) => {
      if (!controller.signal.aborted) setStreamError(caught.message);
    });
    return () => controller.abort();
  }, [runId]);

  const view = useMemo(() => summarize(events), [events]);

  if (!run) return <p className={`page-message ${error ? "error" : "muted"}`}>{error || "Loading…"}</p>;

  const finished = FINISHED_STATUSES.includes(run.status);
  const result = run.result || {};
  const history = result.training?.history || view.history;
  const evaluation = result.evaluation || view.evaluation;
  const model = result.model || view.model;
  const dataset = result.dataset || view.dataset;
  const currentPhase = view.phases.at(-1);

  const act = async (action) => {
    try {
      await action();
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="muted small">
            {run.sandbox_id ? <Link to={`/sandboxes/${run.sandbox_id}`}>sandbox</Link> : "runs"} /
          </p>
          <h1>
            {run.kind} run <StatusBadge status={run.status} />
          </h1>
          <p className="muted small">
            <code>{run.id}</code> · {run.hardware_tier} · started {formatDate(run.started_at || run.created_at)}
            {run.finished_at && run.started_at
              ? ` · took ${formatSeconds((new Date(run.finished_at) - new Date(run.started_at)) / 1000)}`
              : ""}
          </p>
        </div>
        <div className="toolbar">
          {!finished && (
            <button type="button" className="button ghost" onClick={() => act(async () => setRun(await cancelRun(runId)))}>
              <CircleStop size={16} aria-hidden="true" /> Cancel
            </button>
          )}
          {finished && (
            <button
              type="button"
              className="button"
              onClick={() => act(async () => navigate(`/runs/${(await rerun(runId)).id}`))}
            >
              <RotateCcw size={16} aria-hidden="true" /> Run again
            </button>
          )}
        </div>
      </header>

      {error && <p className="error" role="alert">{error}</p>}
      {streamError && !finished && <p className="error">Live updates stopped ({streamError}). Reload to resume.</p>}

      {(run.error || view.failure) && (
        <div className="error-box" role="alert">
          <p>{run.error || view.failure.error}</p>
          {(result.issues || view.failure?.issues || []).length > 0 && (
            <ul>
              {(result.issues || view.failure.issues).map((issue, index) => (
                <li key={index}>
                  {[issue.node_id, issue.field].filter(Boolean).join(".")} {issue.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {!finished && (
        <div className="panel progress-panel" aria-live="polite">
          <p>
            {run.status === "queued" ? "Waiting for a worker…" : currentPhase ? `${currentPhase}…` : "Starting…"}
          </p>
          {view.batch && currentPhase === "training" && (
            <>
              <progress max={view.batch.batches} value={view.batch.batch} />
              <p className="muted small">
                epoch {view.batch.epoch} · batch {view.batch.batch}/{view.batch.batches} · loss{" "}
                {formatMetric(view.batch.loss)} · {view.batch.samples_per_s} samples/s
              </p>
            </>
          )}
        </div>
      )}

      {view.warnings.length > 0 && (
        <ul className="warning-list">
          {view.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      {(dataset || model || view.environment) && (
        <div className="stat-row">
          {dataset && <Stat label="samples" value={formatCount(dataset.samples)} />}
          {dataset?.splits && (
            <Stat
              label="train / val / test"
              value={["train", "val", "test"].map((name) => dataset.splits[name] ?? 0).join(" / ")}
            />
          )}
          {model && <Stat label="parameters (trainable)" value={`${formatCount(model.params)} (${formatCount(model.trainable_params)})`} />}
          {model?.input && <Stat label="input" value={`${model.input.channels} × ${model.input.size.join(" × ")}`} />}
          {result.training && (
            <Stat
              label="best epoch"
              value={`${result.training.best_epoch ?? "—"} of ${result.training.epochs_run}${
                result.training.stopped_reason !== "completed" ? ` (${result.training.stopped_reason.replace("_", " ")})` : ""
              }`}
            />
          )}
          {view.environment && <Stat label="engine" value={`${view.environment.engine} · ${view.environment.template}`} />}
        </div>
      )}

      {run.kind === "train" && history.length > 0 && <TrainingView history={history} monitor={result.training?.monitor} bestEpoch={result.training?.best_epoch ?? history.findLast((e) => e.best)?.epoch} />}
      {run.kind === "train" && Object.keys(evaluation).length > 0 && <EvaluationView evaluation={evaluation} />}
      {run.kind === "sanity" && <SanityView result={run.result} steps={view.sanity} />}
      {run.kind === "profile" && result.profile && <ProfileView profile={result.profile} />}

      {finished && run.result && (
        <details className="section">
          <summary>Full result (JSON)</summary>
          <pre className="code-block">{JSON.stringify(run.result, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}
