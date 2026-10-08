import { Navigate, Outlet, useLocation } from "react-router-dom";

import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return <p className="muted page-message">Loading your workspace…</p>;
  }
  if (!user) {
    return <Navigate to="/auth" replace state={{ from: location }} />;
  }
  return <Outlet />;
}
