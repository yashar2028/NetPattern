import { Activity, Boxes, FlaskConical, Lock, Network, Send } from "lucide-react";

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
  return (
    <div className="home">
      <section className="hero">
        <p className="eyebrow">CNNs for imaging, without the boilerplate</p>
        <h1>Build, train, evaluate and serve CNNs visually.</h1>
        <p className="lede">
          Bring your images (PNG, JPEG, DICOM or NIfTI), shape a network on the canvas, and track
          every run in a reproducible sandbox.
        </p>
      </section>

      <section className="card-grid" aria-label="Workflow">
        {steps.map(({ icon: Icon, title, text }) => (
          <article key={title} className="card">
            <Icon className="card-icon" size={22} aria-hidden="true" />
            <h2>{title}</h2>
            <p>{text}</p>
          </article>
        ))}
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
