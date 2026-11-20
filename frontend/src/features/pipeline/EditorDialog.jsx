import { useLayoutEffect, useRef } from "react";

export default function EditorDialog({ titleId, className = "", fallbackFocus, onClose, children }) {
  const ref = useRef(null);
  useLayoutEffect(() => {
    const dialog = ref.current;
    const doc = dialog.ownerDocument;
    const opener = doc.activeElement;
    const fallback = fallbackFocus?.current;
    dialog.showModal();
    dialog.querySelector("h2")?.focus();
    return () => {
      dialog.close();
      if (opener?.isConnected && opener !== doc.body) opener.focus({ preventScroll: true });
      if ((doc.activeElement !== opener || opener === doc.body) && fallback?.isConnected) {
        fallback.focus({ preventScroll: true });
      }
    };
  }, [fallbackFocus]);

  return (
    <dialog
      ref={ref}
      className={`layer-editor editor-modal ${className}`}
      aria-labelledby={titleId}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || event.defaultPrevented) return;
        event.preventDefault();
        const popover = event.currentTarget.querySelector(":popover-open");
        if (popover) popover.hidePopover();
        else onClose();
      }}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      {children}
    </dialog>
  );
}
