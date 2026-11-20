import { Activity, ArrowRight, Boxes, FlaskConical, Lock, Network, Send } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const steps = [
  {
    icon: Network,
    title: "Build",
    text: "Drag data, transforms, models and training onto a canvas. Compose layers or start from a pretrained ResNet, DenseNet or EfficientNet.",
  },
  {
    icon: Activity,
    title: "Train",
    text: "One click, on the hardware tier you choose. Loss and metrics stream in live.",
  },
  {
    icon: FlaskConical,
    title: "Evaluate",
    text: "Standard metrics immediately; ROC, calibration and Grad-CAM one click away.",
  },
  {
    icon: Send,
    title: "Serve",
    text: "Deploy a model and call it from your own software through the REST API.",
  },
];

const sandboxPoints = [
  {
    icon: Boxes,
    title: "One sandbox per CNN",
    text: "Each model you build lives in its own isolated workspace.",
  },
  {
    icon: Lock,
    title: "Pinned with Nix",
    text: "Python, PyTorch, CUDA and the NetPattern engine are locked, so today's model behaves the same in the future.",
  },
];

export default function HomePage() {
  const { user } = useAuth();
  return (
    <div className="home">
      <section className="hero">
        <p className="eyebrow">A workspace for imaging AI</p>
        <h1>From images to insights, <span>all in one place.</span></h1>
        <p className="lede">
          Build, train and evaluate image models visually. Keep your experiments organized and
          bring your best models into your applications.
        </p>
        <div className="hero-actions">
          <Link className="button primary" to={user ? "/sandboxes" : "/auth?mode=register"}>{user ? "Open your workspace" : "Get started"} <ArrowRight size={17} aria-hidden="true" /></Link>
          <a className="button" href="#how-it-works">See how it works</a>
        </div>
        <p className="muted small">Supports PNG, JPEG, DICOM and NIfTI images.</p>
      </section>

      <section id="how-it-works" className="section">
        <div className="section-heading"><div><p className="eyebrow">Your workflow</p><h2>Everything you need to move forward</h2><p className="muted">From the first experiment to a model you can use.</p></div></div>
        <div className="card-grid" aria-label="Workflow">
        {steps.map(({ icon: Icon, title, text }) => (
          <article key={title} className="card">
            <Icon className="card-icon" size={22} aria-hidden="true" />
            <h2>{title}</h2>
            <p>{text}</p>
          </article>
        ))}
        </div>
      </section>

      <section className="sandbox-band" aria-label="Sandboxes">
        {sandboxPoints.map(({ icon: Icon, title, text }) => (
          <div key={title} className="sandbox-point">
            <Icon className="card-icon" size={20} aria-hidden="true" />
            <div>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
