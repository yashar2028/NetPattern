import { lazy } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import AppLayout from "./components/AppLayout";
import ProtectedRoute from "./components/ProtectedRoute";
import PublicLayout from "./components/PublicLayout";
import { AuthProvider } from "./context/AuthContext";
import ApiKeysPage from "./pages/ApiKeysPage";
import AuthPage from "./pages/AuthPage";
import DatasetPage from "./pages/DatasetPage";
import DatasetsPage from "./pages/DatasetsPage";
import DeploymentsPage from "./pages/DeploymentsPage";
import HomePage from "./pages/HomePage";
import ModelsPage from "./pages/ModelsPage";
import NotFoundPage from "./pages/NotFoundPage";
import SandboxPage from "./pages/SandboxPage";
import SandboxesPage from "./pages/SandboxesPage";
import TeamsPage from "./pages/TeamsPage";
import "./App.css";
import "./workspace.css";

// The canvas and chart libraries load only with the pages that use them.
const PipelinePage = lazy(() => import("./pages/PipelinePage"));
const RunPage = lazy(() => import("./pages/RunPage"));
const DeploymentPage = lazy(() => import("./pages/DeploymentPage"));
const ComparePage = lazy(() => import("./pages/ComparePage"));

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<PublicLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/auth" element={<AuthPage />} />
          </Route>
          <Route element={<ProtectedRoute />}>
            <Route element={<AppLayout />}>
              <Route path="/sandboxes" element={<SandboxesPage />} />
              <Route path="/sandboxes/:sandboxId" element={<SandboxPage />} />
              <Route path="/sandboxes/:sandboxId/pipelines/:pipelineId" element={<PipelinePage />} />
              <Route path="/runs/:runId" element={<RunPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route path="/datasets/:datasetId" element={<DatasetPage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/deployments" element={<DeploymentsPage />} />
              <Route path="/deployments/:deploymentId" element={<DeploymentPage />} />
              <Route path="/api-keys" element={<ApiKeysPage />} />
              <Route path="/teams" element={<TeamsPage />} />
            </Route>
          </Route>
          <Route element={<PublicLayout />}>
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
