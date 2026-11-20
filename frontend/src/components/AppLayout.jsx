import { Suspense } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { Boxes, Database, KeyRound, Layers, LogOut, Rocket, Users } from "lucide-react";

import { useAuth } from "../context/AuthContext";
import BackendStatus from "./BackendStatus";

const navLinks = [
  { to: "/sandboxes", label: "Sandboxes", icon: Boxes },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/models", label: "Models", icon: Layers },
  { to: "/deployments", label: "Deployments", icon: Rocket },
  { to: "/api-keys", label: "API keys", icon: KeyRound },
  { to: "/teams", label: "Teams", icon: Users },
];

export default function AppLayout() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="public-shell app-shell">
      <header className="site-header app-header">
        <NavLink className="brand" to="/sandboxes">
          <img className="brand-mark" src="/favicon.svg" alt="" />
          <span className="brand-name">NetPattern</span>
        </NavLink>
        <nav className="site-nav app-nav" aria-label="Main">
          {navLinks.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) => `site-nav-link${isActive ? " active" : ""}`}
            >
              <link.icon size={16} aria-hidden="true" /> {link.label}
            </NavLink>
          ))}
        </nav>
        <BackendStatus />
        <span className="muted user-email" title={user?.email}>{user?.email}</span>
        <button
          type="button"
          className="button ghost"
          onClick={() => {
            signOut();
            navigate("/auth", { replace: true });
          }}
        >
          <LogOut size={16} aria-hidden="true" /> Sign out
        </button>
      </header>
      <main className="app-main">
        <Suspense fallback={<p className="muted page-message">Loading…</p>}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}
