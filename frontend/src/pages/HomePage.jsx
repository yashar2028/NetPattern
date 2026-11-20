import { Activity, ArrowRight, Boxes, ChartNoAxesCombined, Database, FlaskConical, Layers, LockKeyhole, Network, Rocket, Users } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const steps = [
  { icon: Database, title: "Bring your images", text: "Upload and version your datasets, then profile them to understand their contents." },
  { icon: Network, title: "Design a pipeline", text: "Start from a template or compose your own model, transforms, and training steps on the canvas." },
  { icon: Activity, title: "Train and compare", text: "Run experiments on a hardware tier, follow progress, and compare results across runs." },
  { icon: Rocket, title: "Deploy and use", text: "Register a trained model, try a deployment, and call it from your applications." },
];

const areas = [
  { icon: Boxes, title: "Sandboxes", text: "Organize pipelines and runs in reproducible environments.", to: "/sandboxes" },
  { icon: Database, title: "Datasets", text: "Upload, version, and profile your image data.", to: "/datasets" },
  { icon: Layers, title: "Models", text: "Register, version, export, and deploy trained models.", to: "/models" },
  { icon: Users, title: "Teams", text: "Share sandboxes and datasets with your collaborators.", to: "/teams" },
];

export default function HomePage() {
  const { user } = useAuth();
  return (
    <div className="home">
      <section className="hero home-hero">
        <div className="hero-copy">
          <p className="eyebrow">A workspace for imaging AI</p>
          <h1>From images to insights, <span>all in one place.</span></h1>
          <p className="lede">NetPattern helps you build, train, evaluate and deploy image models visually—with your data, experiments and results organized together.</p>
          <div className="hero-actions">
            <Link className="button primary" to={user ? "/sandboxes" : "/auth?mode=register"}>{user ? "Open your workspace" : "Get started"} <ArrowRight size={17} aria-hidden="true" /></Link>
            <a className="button" href="#how-it-works">Explore the workflow</a>
          </div>
          <p className="muted small">For PNG, JPEG, DICOM and NIfTI imaging workflows.</p>
        </div>
        <div className="hero-diagram" aria-label="Workflow: data, model, training and results">
          <p className="eyebrow">Your workflow</p>
          <div><Database size={18} aria-hidden="true" /><span>Image data</span><ArrowRight size={16} aria-hidden="true" /></div>
          <div><Network size={18} aria-hidden="true" /><span>Visual pipeline</span><ArrowRight size={16} aria-hidden="true" /></div>
          <div><Activity size={18} aria-hidden="true" /><span>Training runs</span><ArrowRight size={16} aria-hidden="true" /></div>
          <div><ChartNoAxesCombined size={18} aria-hidden="true" /><span>Results &amp; deployment</span></div>
        </div>
      </section>

      <section id="how-it-works" className="section home-section">
        <div className="section-heading"><div><p className="eyebrow">How it works</p><h2>One place for your model lifecycle</h2><p className="muted">Move from your first dataset to a model you can use.</p></div></div>
        <div className="home-step-grid">
          {steps.map(({ icon: Icon, title, text }, index) => (
            <article key={title} className="card home-step">
              <div className="home-step-top"><Icon className="card-icon" size={22} aria-hidden="true" /><span>0{index + 1}</span></div>
              <h3>{title}</h3><p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="explore" className="section home-section">
        <div className="section-heading"><div><p className="eyebrow">Explore NetPattern</p><h2>Find the right place to start</h2><p className="muted">Every part of your workflow has a home.</p></div></div>
        <div className="home-area-grid">
          {areas.map(({ icon: Icon, title, text, to }) => (
            <Link key={title} className="card card-link home-area" to={to}>
              <Icon className="card-icon" size={20} aria-hidden="true" />
              <div><h3>{title}</h3><p>{text}</p></div>
              <ArrowRight className="home-area-arrow" size={18} aria-hidden="true" />
            </Link>
          ))}
        </div>
      </section>

      <section id="reproducibility" className="home-repro">
        <div className="feature-icon"><LockKeyhole size={22} aria-hidden="true" /></div>
        <div><p className="eyebrow">Built for reproducibility</p><h2>Experiments you can return to</h2><p>Each sandbox pins its engine and software environment, helping you keep track of how your model was built.</p></div>
        <FlaskConical className="repro-icon" size={40} aria-hidden="true" />
      </section>
      <section className="home-cta"><div><h2>Ready to start exploring?</h2><p>Set up your workspace and take your first step.</p></div><Link className="button primary" to={user ? "/sandboxes" : "/auth?mode=register"}>{user ? "Open workspace" : "Create an account"} <ArrowRight size={16} aria-hidden="true" /></Link></section>
    </div>
  );
}
