import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { errorMessage } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function AuthPage() {
  const { user, signIn, signUp } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mode, setMode] = useState("login");
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
    <section className="panel auth-panel">
      <h1>{mode === "login" ? "Sign in" : "Create an account"}</h1>
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
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="button primary" type="submit" disabled={busy}>
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
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
    </section>
  );
}
