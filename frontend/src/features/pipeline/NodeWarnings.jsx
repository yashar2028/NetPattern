import { useId, useRef } from "react";
import { CircleAlert } from "lucide-react";

export default function NodeWarnings({ label, warnings, onWarning }) {
  const id = useId();
  const trigger = useRef(null);
  const popover = useRef(null);
  if (!warnings.length) return null;

  const name = `${label}: ${warnings.length} warning${warnings.length === 1 ? "" : "s"}`;
  const close = () => {
    popover.current.hidePopover();
    trigger.current.focus({ preventScroll: true });
  };

  return (
    <div
      className="node-warnings nodrag nopan nowheel"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !popover.current.matches(":popover-open")) return;
        event.preventDefault();
        event.stopPropagation();
        close();
      }}
    >
      <button
        ref={trigger}
        type="button"
        className="node-warning-trigger"
        popoverTarget={id}
        aria-label={name}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") event.stopPropagation();
        }}
      >
        <CircleAlert size={14} strokeWidth={1.6} aria-hidden="true" />
        <span aria-hidden="true">{warnings.length}</span>
      </button>
      <div
        ref={popover}
        id={id}
        popover="auto"
        className="node-warning-popover"
        role="group"
        aria-labelledby={`${id}-title`}
        onKeyDown={(event) => {
          if (event.key !== "Escape") event.stopPropagation();
        }}
      >
        <strong id={`${id}-title`}>{name}</strong>
        <ul>
          {warnings.map((warning, index) => (
            <li key={index}>
              {onWarning && warning.action ? (
                <button type="button" className="button ghost" onClick={() => { close(); onWarning(warning); }}>
                  {warning.message}
                  <span className="small">{warning.action}</span>
                </button>
              ) : warning.message}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
