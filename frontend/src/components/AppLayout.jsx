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

function dismissDropdown(event) {
  if (event.key !== "Escape" || !event.currentTarget.open) return;
  event.preventDefault();
  event.currentTarget.open = false;
  event.currentTarget.querySelector("summary").focus();
}

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
        <nav className="app-nav" aria-label="Main">
          <details className="compact-disclosure header-dropdown" name="app-header-dropdowns" onKeyDown={dismissDropdown}>
            <summary>Workspace</summary>
            <div className="dropdown-content">
              {navLinks.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={({ isActive }) => `site-nav-link${isActive ? " active" : ""}`}
                  onClick={(event) => {
                    const disclosure = event.currentTarget.closest("details");
                    disclosure.querySelector("summary").focus();
                    disclosure.open = false;
                  }}
                >
                  <link.icon size={16} aria-hidden="true" /> {link.label}
                </NavLink>
              ))}
            </div>
          </details>
        </nav>
        <BackendStatus />
        <details className="compact-disclosure header-dropdown account-dropdown" name="app-header-dropdowns" onKeyDown={dismissDropdown}>
          <summary>Account</summary>
          <div className="dropdown-content">
            <p className="muted small">{user?.email}</p>
            <button type="button" className="button ghost" onClick={() => {
              signOut();
              navigate("/auth", { replace: true });
            }}>
              <LogOut size={16} aria-hidden="true" /> Sign out
            </button>
          </div>
        </details>
      </header>
      <main className="app-main">
        <Suspense fallback={<p className="muted page-message">Loading…</p>}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}
