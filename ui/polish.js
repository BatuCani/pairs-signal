// News Gate visual polish: cursor-following glint on primary buttons. Effect only, no behavior changes.
(() => {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  document.addEventListener("pointermove", (e) => {
    const btn = e.target.closest && e.target.closest(".btn");
    if (!btn) return;
    const r = btn.getBoundingClientRect();
    btn.style.setProperty("--mx", `${e.clientX - r.left}px`);
    btn.style.setProperty("--my", `${e.clientY - r.top}px`);
  }, { passive: true });
  document.addEventListener("pointerout", (e) => {
    const btn = e.target.closest && e.target.closest(".btn");
    if (btn && !btn.contains(e.relatedTarget)) {
      btn.style.removeProperty("--mx");
      btn.style.removeProperty("--my");
    }
  }, { passive: true });
})();
