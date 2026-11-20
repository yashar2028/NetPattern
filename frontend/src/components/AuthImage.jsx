import { useEffect, useState } from "react";

import { runFile } from "../api/runsApi";

/** An image from a run's files, fetched with the user's token. */
export default function AuthImage({ runId, path, alt }) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    let active = true;
    let objectUrl = null;
    runFile(runId, path)
      .then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {});
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [runId, path]);
  return url ? <img src={url} alt={alt} /> : <div className="image-placeholder" aria-label={alt} />;
}
