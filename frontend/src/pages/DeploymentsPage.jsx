import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { errorMessage } from "../api/client";
import { listDeployments } from "../api/modelsApi";
import StatusBadge from "../components/StatusBadge";
import { formatDate } from "../utils/format";

export default function DeploymentsPage() {
  const [deployments, setDeployments] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    listDeployments().then(setDeployments).catch((caught) => setError(errorMessage(caught)));
  }, []);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Deployments</h1>
          <p className="muted">Models you can call over HTTP. Deploy one from the Models page.</p>
        </div>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {deployments === null && !error && <p className="muted">Loading deployments…</p>}
      {deployments?.length === 0 && <div className="empty-state"><strong>No deployments yet</strong><p>Deploy a registered model to make it available through the API.</p><Link className="button" to="/models">Explore models</Link></div>}
      {deployments?.length > 0 && (
        <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Mode</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {deployments.map((deployment) => (
              <tr key={deployment.id}>
                <td>
                  <Link to={`/deployments/${deployment.id}`}>{deployment.name}</Link>
                </td>
                <td>
                  <StatusBadge status={deployment.status} />
                </td>
                <td>{deployment.mode}</td>
                <td>{formatDate(deployment.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </div>
  );
}
