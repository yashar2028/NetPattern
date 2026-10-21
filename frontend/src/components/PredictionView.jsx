import { useEffect, useMemo, useRef, useState } from "react";

import { formatMetric } from "../utils/format";

// Mask colors per class index (0 is usually background and stays transparent).
const MASK_COLORS = [
  [0, 0, 0, 0],
  [42, 120, 214, 150],
  [235, 104, 52, 150],
  [27, 175, 122, 150],
  [160, 90, 200, 150],
  [220, 180, 40, 150],
];

function useObjectUrl(file) {
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => url && URL.revokeObjectURL(url), [url]);
  return url;
}

function MaskOverlay({ file, maskBase64 }) {
  const canvas = useRef(null);
  const imageUrl = useObjectUrl(file);
  useEffect(() => {
    const mask = new Image();
    mask.onload = () => {
      const element = canvas.current;
      if (!element) return;
      element.width = mask.width;
      element.height = mask.height;
      const context = element.getContext("2d");
      context.drawImage(mask, 0, 0);
      const pixels = context.getImageData(0, 0, mask.width, mask.height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const color = MASK_COLORS[pixels.data[i] % MASK_COLORS.length];
        pixels.data.set(color, i);
      }
      context.putImageData(pixels, 0, 0);
    };
    mask.src = `data:image/png;base64,${maskBase64}`;
  }, [maskBase64]);
  return (
    <div className="overlay">
      {imageUrl && <img src={imageUrl} alt="input" />}
      <canvas ref={canvas} aria-label="predicted mask" />
    </div>
  );
}

function BoxOverlay({ file, detections }) {
  const imageUrl = useObjectUrl(file);
  const [size, setSize] = useState(null);
  return (
    <div className="overlay">
      {imageUrl && (
        <img
          src={imageUrl}
          alt="input"
          onLoad={(event) => setSize([event.target.naturalWidth, event.target.naturalHeight])}
        />
      )}
      {size && (
        <svg viewBox={`0 0 ${size[0]} ${size[1]}`} aria-label="predicted boxes">
          {detections.map((d, index) => {
            const [x1, y1, x2, y2] = d.box;
            return (
              <g key={index}>
                <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} className="box" />
                <text x={x1 + 3} y={y1 + 14} className="box-label">
                  {d.label} {formatMetric(d.score)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function HeatmapOverlay({ explanation }) {
  const canvas = useRef(null);
  useEffect(() => {
    const heat = new Image();
    heat.onload = () => {
      const element = canvas.current;
      if (!element) return;
      element.width = heat.width;
      element.height = heat.height;
      const context = element.getContext("2d");
      context.drawImage(heat, 0, 0);
      const pixels = context.getImageData(0, 0, heat.width, heat.height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const strength = pixels.data[i];
        pixels.data.set([235, 104, 52, Math.round(strength * 0.65)], i);
      }
      context.putImageData(pixels, 0, 0);
    };
    heat.src = `data:image/png;base64,${explanation.heatmap_png_base64}`;
  }, [explanation]);
  return (
    <figure className="stack">
      <div className="overlay pixelated">
        <img src={`data:image/png;base64,${explanation.input_png_base64}`} alt="image as the model saw it" />
        <canvas ref={canvas} aria-label="Grad-CAM heatmap" />
      </div>
      <figcaption className="muted small">
        Grad-CAM for “{explanation.class}” on layer <code>{explanation.layer}</code>: orange marks what raised this class.
      </figcaption>
    </figure>
  );
}

/** One prediction, shown the way its task reads best; the raw JSON is always one click away. */
export default function PredictionView({ task, output, file }) {
  let body = null;
  if (task === "classification.single_label") {
    body = (
      <table className="table compact">
        <tbody>
          {output.top_k.map((item) => (
            <tr key={item.label}>
              <th scope="row">{item.label}</th>
              <td className="number">{formatMetric(item.score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  } else if (task === "classification.multi_label") {
    body = <p>{output.labels.length ? output.labels.map((l) => l.label).join(", ") : "no label above the threshold"}</p>;
  } else if (task === "regression") {
    body = (
      <table className="table compact">
        <tbody>
          {Object.entries(output.values).map(([name, value]) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td className="number">{formatMetric(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  } else if (task === "segmentation.semantic") {
    body = <MaskOverlay file={file} maskBase64={output.mask_png_base64} />;
  } else if (task === "detection.bbox") {
    body = <BoxOverlay file={file} detections={output.detections} />;
  }
  return (
    <div className="stack">
      {body}
      {output.explanation?.heatmap_png_base64 && <HeatmapOverlay explanation={output.explanation} />}
      {output.explanation?.error && <p className="muted small">No explanation: {output.explanation.error}</p>}
      <details>
        <summary>JSON</summary>
        <pre className="code-block">
          {JSON.stringify(
            {
              ...output,
              mask_png_base64: output.mask_png_base64 ? "…" : undefined,
              explanation: output.explanation ? { ...output.explanation, heatmap_png_base64: "…", input_png_base64: "…" } : undefined,
            },
            null,
            2
          )}
        </pre>
      </details>
    </div>
  );
}
