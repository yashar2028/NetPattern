import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Database, Plus } from "lucide-react";

import { errorMessage } from "../api/client";
import { createDataset, listDatasets } from "../api/datasetsApi";
import { formatDate } from "../utils/format";

export default function DatasetsPage() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState(null);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    listDatasets().then(setDatasets).catch((caught) => setError(errorMessage(caught)));
  }, []);

  const create = async (event) => {
    event.preventDefault();
    try {
      const dataset = await createDataset({ name });
      navigate(`/datasets/${dataset.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Datasets</h1>
          <p className="muted">Upload a zip of your image folder. Every upload becomes a version that never changes.</p>
        </div>
      </header>
      <form className="panel inline-form" onSubmit={create}>
        <label className="field grow">
          <span>New dataset</span>
          <input required placeholder="e.g. pets" value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <button className="button primary" type="submit">
          <Plus size={16} aria-hidden="true" /> Create
        </button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {datasets?.length === 0 && <p className="muted">No datasets yet.</p>}
      <div className="card-grid">
        {datasets?.map((dataset) => (
          <Link key={dataset.id} className="card card-link" to={`/datasets/${dataset.id}`}>
            <Database className="card-icon" size={20} aria-hidden="true" />
            <h2>{dataset.name}</h2>
            <p className="muted small">created {formatDate(dataset.created_at)}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
