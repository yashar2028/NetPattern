import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "../context/AuthContext";
import BackendStatus from "./BackendStatus";

export default function PublicLayout() {
  const { user } = useAuth();
  const navLinks = [
    { to: "/", label: "Home", end: true },
    user ? { to: "/sandboxes", label: "Open workspace" } : { to: "/auth", label: "Sign in" },
  ];
  return (
    <div className="public-shell">
      <header className="site-header">
        <NavLink className="brand" to="/">
          <img className="brand-mark" src="/favicon.svg" alt="" />
          <span className="brand-name">NetPattern</span>
        </NavLink>

        <nav className="site-nav" aria-label="Main">
          {navLinks.slice(0, 1).map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => `site-nav-link${isActive ? " active" : ""}`}
            >
              {link.label}
            </NavLink>
          ))}
          <a className="site-nav-link" href="/#how-it-works">How it works</a>
          <a className="site-nav-link" href="/#explore">Explore</a>
          <NavLink to={navLinks[1].to} className={({ isActive }) => `site-nav-link${isActive ? " active" : ""}`}>{navLinks[1].label}</NavLink>
        </nav>

        <BackendStatus />
      </header>

      <main className="site-main">
        <Outlet />
      </main>

      <footer className="site-footer">
        <span>NetPattern · A workspace for image-based deep learning</span>
        <div><a href="/#how-it-works">Workflow</a><a href="/#explore">Explore</a><NavLink to="/auth">Account</NavLink></div>
      </footer>
    </div>
  );
}
