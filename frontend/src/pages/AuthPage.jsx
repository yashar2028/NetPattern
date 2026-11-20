import { useEffect, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Boxes, Database, Network } from "lucide-react";

import { getAgreement } from "../api/authApi";
import { errorMessage } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { registrationPayload } from "../features/registration";

export default function AuthPage() {
  const { user, signIn, signUp } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const mode = new URLSearchParams(location.search).get("mode") === "register" ? "register" : "login";
  const [form, setForm] = useState({ email: "", password: "", full_name: "", purpose: "", confirm: "", accepted: false });
  const [agreement, setAgreement] = useState(null);
  const [agreementError, setAgreementError] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (mode !== "register") return undefined;
    let active = true;
    getAgreement().then((data) => {
      if (active) setAgreement(data);
    }).catch((caught) => {
      if (active) setAgreementError(errorMessage(caught));
    });
    return () => { active = false; };
  }, [mode]);

  const from = location.state?.from;
  const target = from ? `${from.pathname}${from.search || ""}${from.hash || ""}` : "/sandboxes";
  if (user) return <Navigate to={target} replace />;

  const switchMode = (next) => {
    setError("");
    setAgreementError("");
    setAgreement(null);
    setForm((current) => ({ ...current, password: "", confirm: "", accepted: false }));
    navigate(next === "register" ? "/auth?mode=register" : "/auth", { replace: true, state: location.state });
  };

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "login") {
        await signIn({ email: form.email, password: form.password });
      } else {
        await signUp(registrationPayload(form, agreement));
      }
      navigate(target, { replace: true });
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const update = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));

  return (
    <div className="auth-layout">
      <section className="auth-intro">
        <p className="eyebrow">Welcome to NetPattern</p>
        <h1>A clearer path from images to models.</h1>
        <p>Bring your imaging experiments together, from your first dataset to a deployed model.</p>
        <div className="auth-benefits">
          <div><Database size={19} aria-hidden="true" /><span><strong>Organize your data</strong><small>Keep versions of your datasets in one place.</small></span></div>
          <div><Network size={19} aria-hidden="true" /><span><strong>Build visually</strong><small>Turn your model ideas into editable pipelines.</small></span></div>
          <div><Boxes size={19} aria-hidden="true" /><span><strong>Keep experiments reproducible</strong><small>Each sandbox pins its software environment.</small></span></div>
        </div>
      </section>
      <section className="panel auth-panel">
        <div className="auth-switch" aria-label="Account options">
          <button type="button" className={mode === "login" ? "selected" : ""} aria-pressed={mode === "login"} disabled={busy} onClick={() => switchMode("login")}>Sign in</button>
          <button type="button" className={mode === "register" ? "selected" : ""} aria-pressed={mode === "register"} disabled={busy} onClick={() => switchMode("register")}>Create account</button>
        </div>
        <div key={mode} className="auth-content">
          <h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2>
          <p className="muted">{mode === "login" ? "Sign in to pick up where you left off." : "Tell us a little about your work to get started."}</p>
          <form className="stack" onSubmit={submit}>
            {mode === "register" && (
              <label className="field">
                <span>Name (optional)</span>
                <input value={form.full_name} onChange={update("full_name")} autoComplete="name" maxLength={120} />
              </label>
            )}
            <label className="field">
              <span>Email</span>
              <input type="email" required maxLength={320} value={form.email} onChange={update("email")} autoComplete="email" placeholder="you@example.com" />
            </label>
            {mode === "register" && (
              <label className="field">
                <span>What will you use NetPattern for?</span>
                <input required maxLength={250} value={form.purpose} onChange={update("purpose")} placeholder="e.g. medical imaging research" />
              </label>
            )}
            <label className="field">
              <span>Password</span>
              <input type="password" required minLength={mode === "register" ? 8 : 1} maxLength={200} value={form.password} onChange={update("password")} autoComplete={mode === "login" ? "current-password" : "new-password"} />
              {mode === "register" && <small className="muted">Use at least 8 characters.</small>}
            </label>
            {mode === "register" && (
              <>
                <label className="field">
                  <span>Confirm password</span>
                  <input type="password" required value={form.confirm} onChange={update("confirm")} autoComplete="new-password" />
                </label>
                <div className="agreement-box">
                  <strong>Terms and privacy placeholders</strong>
                  <p className="small muted">These are placeholder documents, not final legal terms. Your acceptance and the version are recorded.</p>
                  {agreement ? (
                    <>
                      <details><summary>Read terms and agreement</summary><p>{agreement.terms}</p></details>
                      <details><summary>Read privacy policy</summary><p>{agreement.privacy}</p></details>
                      <p className="small muted">Version: {agreement.version}</p>
                      <label className="agreement-check">
                        <input type="checkbox" required checked={form.accepted} onChange={(event) => setForm((current) => ({ ...current, accepted: event.target.checked }))} />
                        <span>I have read and accept these placeholder documents.</span>
                      </label>
                    </>
                  ) : <p className={agreementError ? "error small" : "muted small"} role={agreementError ? "alert" : "status"}>{agreementError ? `Unable to load agreement: ${agreementError}. Switch tabs and try again.` : "Loading agreement…"}</p>}
                </div>
              </>
            )}
            {error && <p className="error" role="alert">{error}</p>}
            <button className="button primary" type="submit" disabled={busy || (mode === "register" && !agreement)}>
              {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"} <ArrowRight size={16} aria-hidden="true" />
            </button>
          </form>
        </div>
        <Link className="auth-home-link" to="/">Learn more about NetPattern</Link>
      </section>
    </div>
  );
}
