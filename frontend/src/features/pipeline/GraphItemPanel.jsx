import { useEffect, useRef } from "react";

import { canvasPanelPlacement } from "./useCanvasLayout";

export default function GraphItemPanel({
  id = "graph-item-settings", anchor, fallback, workspace, title, closeLabel = "Close settings",
  onDismiss, focusIssueField, children,
}) {
  const ref = useRef(null);
  const heading = useRef(null);

  useEffect(() => {
    const panel = ref.current;
    const doc = panel.ownerDocument;
    const viewport = doc.defaultView;
    const trigger = anchor || fallback?.current;
    const surface = workspace?.current;
    const place = () => {
      const { clientWidth: width, clientHeight: height } = doc.documentElement;
      const { top, maxWidth, maxHeight } = canvasPanelPlacement(surface?.getBoundingClientRect().top ?? 0, width, height);
      panel.style.maxWidth = `${maxWidth}px`;
      panel.style.top = `${top}px`;
      panel.style.maxHeight = `${maxHeight}px`;
    };
    place();
    if (trigger instanceof viewport.HTMLElement) panel.showPopover({ source: trigger });
    else panel.showPopover();
    if (!focusIssueField?.()) heading.current.focus({ preventScroll: true });
    const resize = new ResizeObserver(place);
    if (surface) resize.observe(surface);
    viewport.addEventListener("resize", place);
    viewport.addEventListener("scroll", place);
    return () => {
      resize.disconnect();
      viewport.removeEventListener("resize", place);
      viewport.removeEventListener("scroll", place);
    };
  }, [anchor, fallback, workspace, focusIssueField]);

  return (
    <section ref={ref} id={id} className="graph-item-panel stack" popover="auto"
      aria-labelledby={`${id}-title`}
      onToggle={(event) => {
        if (event.newState === "closed" && !ref.current?.matches(":popover-open")) onDismiss(false);
      }}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || event.defaultPrevented) return;
        event.preventDefault();
        event.stopPropagation();
        onDismiss(true);
      }}>
      <div className="graph-panel-heading">
        <h3 ref={heading} id={`${id}-title`} tabIndex={-1}>{title}</h3>
        <button type="button" className="button ghost" onClick={() => onDismiss(true)}>{closeLabel}</button>
      </div>
      {children}
    </section>
  );
}
