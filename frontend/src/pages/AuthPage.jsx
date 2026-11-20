import { useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Boxes, Database, Network } from "lucide-react";

import { errorMessage } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function AuthPage() {
  const { user, signIn, signUp } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mode, setMode] = useState(() => new URLSearchParams(location.search).get("mode") === "register" ? "register" : "login");
  const [form, setForm] = useState({ email: "", password: "", full_name: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const target = location.state?.from?.pathname || "/sandboxes";
  if (user) return <Navigate to={target} replace />;

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (mode === "login") {
        await signIn({ email: form.email, password: form.password });
      } else {
        await signUp({ email: form.email, password: form.password, full_name: form.full_name || null });
      }
      navigate(target, { replace: true });
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const update = (key) => (event) => setForm({ ...form, [key]: event.target.value });

  return (
    <div className="auth-layout">
    <section className="auth-intro">
      <p className="eyebrow">Welcome to NetPattern</p>
      <h1>Your ideas deserve a better workspace.</h1>
      <p>Bring your imaging experiments together, from your first dataset to a deployed model.</p>
      <div className="auth-benefits">
        <div><Database size={19} aria-hidden="true" /><span><strong>Organize your data</strong><small>Keep versions of your datasets in one place.</small></span></div>
        <div><Network size={19} aria-hidden="true" /><span><strong>Build visually</strong><small>Turn your model ideas into editable pipelines.</small></span></div>
        <div><Boxes size={19} aria-hidden="true" /><span><strong>Keep experiments reproducible</strong><small>Each sandbox pins its software environment.</small></span></div>
      </div>
    </section>
    <section className="panel auth-panel">
      <p className="eyebrow">Your workspace awaits</p>
      <h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2>
      <p className="muted">{mode === "login" ? "Sign in to pick up where you left off." : "Get started with your own workspace."}</p>
      <form className="stack" onSubmit={submit}>
        {mode === "register" && (
          <label className="field">
            <span>Name (optional)</span>
            <input value={form.full_name} onChange={update("full_name")} autoComplete="name" />
          </label>
        )}
        <label className="field">
          <span>Email</span>
          <input type="email" required value={form.email} onChange={update("email")} autoComplete="email" />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            required
            minLength={mode === "register" ? 8 : 1}
            value={form.password}
            onChange={update("password")}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
          />
          {mode === "register" && <small className="muted">Use at least 8 characters.</small>}
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="button primary" type="submit" disabled={busy}>
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"} <ArrowRight size={16} aria-hidden="true" />
        </button>
      </form>
      <button
        type="button"
        className="link-button"
        onClick={() => {
          setMode(mode === "login" ? "register" : "login");
          setError("");
        }}
      >
        {mode === "login" ? "No account yet? Create one" : "Already have an account? Sign in"}
      </button>
      <Link className="auth-home-link" to="/">Learn more about NetPattern</Link>
    </section>
    </div>
  );
}
