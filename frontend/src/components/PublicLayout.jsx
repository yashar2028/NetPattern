import { NavLink, Outlet } from "react-router-dom";

import BackendStatus from "./BackendStatus";

const navLinks = [
  { to: "/", label: "Home", end: true },
  { to: "/auth", label: "Sign in" },
];

export default function PublicLayout() {
  return (
    <div className="public-shell">
      <header className="site-header">
        <NavLink className="brand" to="/">
          <img className="brand-mark" src="/favicon.svg" alt="" />
          <span className="brand-name">NetPattern</span>
        </NavLink>

        <nav className="site-nav" aria-label="Main">
          {navLinks.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => `site-nav-link${isActive ? " active" : ""}`}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <BackendStatus />
      </header>

      <main className="site-main">
        <Outlet />
      </main>

      <footer className="site-footer">
        <span>NetPattern · Phase 0 skeleton</span>
      </footer>
    </div>
  );
}
