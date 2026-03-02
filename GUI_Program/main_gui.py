import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import random
import os
import ast

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# ── Colour palette ───────────────────────────────────────────────
BG          = "#1e2130"
PANEL_BG    = "#252a3a"
CANVAS_BG   = "#0d1117"
ACCENT      = "#2979ff"
ACCENT_DARK = "#1a56cc"
RED         = "#f44336"
GREEN       = "#00e676"
YELLOW      = "#ffd740"
TEXT_MAIN   = "#e8eaf6"
TEXT_DIM    = "#7986cb"
BORDER      = "#3d4466"
# ─────────────────────────────────────────────────────────────────


def parse_onnx_class_names(model_path: str) -> list[str]:
    """
    Read class names from YOLOv8 .onnx metadata.
    Returns a list of class name strings, or [] on failure.
    """
    if not HAS_ORT:
        return []
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        # YOLOv8 stores names as  "{'0': 'cat', '1': 'dog', ...}"
        raw = meta.get("names", "")
        if raw:
            parsed = ast.literal_eval(raw)          # dict  {0: 'cat', ...}
            if isinstance(parsed, dict):
                return [parsed[k] for k in sorted(parsed.keys())]
            if isinstance(parsed, list):
                return parsed
    except Exception:
        pass
    return []


class RealTimeDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLOv8 · Real-Time Object Detection")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.running      = False
        self.fps_val      = 0.0
        self.class_names: list[str] = []          # filled after browse
        self.count_labels: dict[str, tk.Label] = {}

        self._build_ui()

    # ── helpers ───────────────────────────────────────────────────
    def _card(self, parent, **kw) -> tk.Frame:
        kw.setdefault("bg", PANEL_BG)
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", BORDER)
        return tk.Frame(parent, **kw)

    # ── build static UI ───────────────────────────────────────────
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG, padx=14, pady=12)
        outer.pack()

        # ── Left : video canvas ───────────────────────────────────
        canvas_wrap = self._card(outer, bg=CANVAS_BG)
        canvas_wrap.grid(row=0, column=0, padx=(0, 14), pady=0, sticky="n")

        self.canvas = tk.Canvas(canvas_wrap, width=340, height=260,
                                bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(padx=2, pady=2)

        # decorative grid lines
        self.canvas.create_line(170, 0, 170, 260, fill="#1e3a5f", width=1)
        self.canvas.create_line(0, 130, 340, 130, fill="#1e3a5f", width=1)
        # corner brackets
        for cx, cy, dx, dy in [(8,8,1,1),(332,8,-1,1),(8,252,1,-1),(332,252,-1,-1)]:
            self.canvas.create_line(cx, cy, cx+dx*18, cy,   fill=ACCENT, width=2)
            self.canvas.create_line(cx, cy, cx,       cy+dy*18, fill=ACCENT, width=2)

        self.cv_label = self.canvas.create_text(
            170, 130, text="Video  RealTime",
            fill="#3a5f8a", font=("Consolas", 16, "bold"), justify="center")

        self.status_dot = self.canvas.create_oval(10, 8, 22, 20,
                                                   fill="#333333", outline="")

        # ── Right : control panel ─────────────────────────────────
        right = tk.Frame(outer, bg=BG)
        right.grid(row=0, column=1, sticky="n")

        # title strip
        title_bar = self._card(right)
        title_bar.pack(fill="x", pady=(0, 10))
        tk.Label(title_bar, text="  DETECTION  PANEL",
                 bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Consolas", 9, "bold"), pady=5).pack(side="left")

        # ── model selector ────────────────────────────────────────
        model_card = self._card(right)
        model_card.pack(fill="x", pady=(0, 10), ipady=4, ipadx=8)

        tk.Label(model_card, text="MODEL", bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=8, pady=(6, 2))

        mrow = tk.Frame(model_card, bg=PANEL_BG)
        mrow.pack(fill="x", padx=8, pady=(0, 6))

        self.model_var = tk.StringVar(value="")
        entry_kw = dict(bg="#0d1117", fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                        relief="flat", font=("Consolas", 10),
                        highlightthickness=1, highlightbackground=BORDER,
                        highlightcolor=ACCENT)
        self.model_entry = tk.Entry(mrow, textvariable=self.model_var,
                                    width=15, **entry_kw)
        self.model_entry.pack(side="left", ipady=4)
        self.model_entry.insert(0, "no model loaded")
        self.model_entry.config(fg=TEXT_DIM)

        browse_btn = tk.Button(mrow, text="browse", command=self._browse_model,
                               bg="#2e3450", fg=TEXT_MAIN,
                               activebackground=ACCENT, activeforeground="white",
                               relief="flat", font=("Consolas", 9),
                               padx=8, pady=4, cursor="hand2",
                               highlightthickness=1, highlightbackground=BORDER)
        browse_btn.pack(side="left", padx=(6, 0))

        # ── RUN / STOP button ─────────────────────────────────────
        self.run_btn = tk.Button(right, text="▶  RUN",
                                  command=self._toggle_run,
                                  bg=ACCENT, fg="white",
                                  activebackground=ACCENT_DARK, activeforeground="white",
                                  font=("Consolas", 14, "bold"),
                                  width=14, height=1, relief="flat",
                                  cursor="hand2", pady=8)
        self.run_btn.pack(pady=(0, 10))

        # ── Object count card (dynamic rows inserted here) ────────
        self.obj_card = self._card(right)
        self.obj_card.pack(fill="x", pady=(0, 10), ipadx=8, ipady=4)

        self.obj_title = tk.Label(self.obj_card, text="OBJECT  COUNT",
                                   bg=PANEL_BG, fg=TEXT_DIM,
                                   font=("Consolas", 8, "bold"))
        self.obj_title.grid(row=0, column=0, columnspan=4,
                             sticky="w", padx=8, pady=(6, 4))

        self.obj_divider = tk.Frame(self.obj_card, bg=BORDER, height=1)
        self.obj_divider.grid(row=1, column=0, columnspan=4,
                               sticky="ew", padx=8, pady=(0, 4))

        # placeholder shown before any model is loaded
        self.obj_placeholder = tk.Label(
            self.obj_card,
            text="  — browse a model to see classes —",
            bg=PANEL_BG, fg=TEXT_DIM,
            font=("Consolas", 9), pady=6)
        self.obj_placeholder.grid(row=2, column=0, columnspan=4,
                                   sticky="w", padx=8, pady=(0, 4))

        # ── FPS row ───────────────────────────────────────────────
        fps_card = self._card(right)
        fps_card.pack(fill="x", pady=(0, 10), ipadx=8, ipady=4)

        fps_inner = tk.Frame(fps_card, bg=PANEL_BG)
        fps_inner.pack(fill="x", padx=8, pady=4)

        tk.Label(fps_inner, text="FPS", bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Consolas", 9, "bold")).pack(side="left")

        self.fps_label = tk.Label(fps_inner, text="---.--",
                                   bg=PANEL_BG, fg=YELLOW,
                                   font=("Consolas", 13, "bold"))
        self.fps_label.pack(side="left", padx=10)

        bar_bg = tk.Frame(fps_inner, bg=BORDER, width=80, height=8)
        bar_bg.pack(side="left")
        bar_bg.pack_propagate(False)
        self.fps_bar = tk.Frame(bar_bg, bg=YELLOW, height=8)
        self.fps_bar.place(x=0, y=0, height=8, width=0)

        # ── EXIT button ───────────────────────────────────────────
        exit_btn = tk.Button(right, text="EXIT",
                              command=self.root.quit,
                              bg=RED, fg="white",
                              activebackground="#b71c1c", activeforeground="white",
                              font=("Consolas", 11, "bold"),
                              width=8, relief="flat", cursor="hand2", pady=5)
        exit_btn.pack(anchor="e", pady=(4, 0))

    # ── browse & load class names ─────────────────────────────────
    def _browse_model(self):
        if self.running:
            return                                  # don't switch mid-run

        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        path = filedialog.askopenfilename(
            initialdir=models_dir,
            title="Select ONNX model",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")])

        if not path:
            return

        # update entry
        self.model_var.set(os.path.basename(path))
        self.model_entry.config(fg=TEXT_MAIN)

        # read classes in background thread (model can be large)
        def _load():
            names = parse_onnx_class_names(path)
            self.root.after(0, self._on_classes_loaded, names)

        threading.Thread(target=_load, daemon=True).start()

    def _on_classes_loaded(self, names: list[str]):
        self.class_names = names if names else []

        # remove old dynamic rows
        for lbl in self.count_labels.values():
            lbl.destroy()
        self.count_labels.clear()

        # remove placeholder
        self.obj_placeholder.grid_remove()

        if not self.class_names:
            # fallback: couldn't read names
            self.obj_placeholder.config(
                text="  ⚠  no class names found in metadata")
            self.obj_placeholder.grid()
            return

        # create one row per class  (starting at grid row 2)
        ICONS = ["●", "◆", "▲", "★", "■", "✦", "◉", "▶"]
        for i, cls in enumerate(self.class_names):
            row = i + 2
            icon = ICONS[i % len(ICONS)]
            tk.Label(self.obj_card,
                     text=f"  {icon}  {cls}",
                     bg=PANEL_BG, fg=GREEN,
                     font=("Consolas", 11)).grid(
                     row=row, column=0, sticky="w", padx=8, pady=3)

            tk.Label(self.obj_card, text="=",
                     bg=PANEL_BG, fg=TEXT_DIM,
                     font=("Consolas", 11)).grid(
                     row=row, column=1, padx=8)

            badge = tk.Label(self.obj_card, text="0",
                              bg="#0d1117", fg=RED,
                              font=("Consolas", 13, "bold"),
                              width=4, anchor="center",
                              highlightthickness=1, highlightbackground=BORDER)
            badge.grid(row=row, column=2, padx=(0, 10), pady=3, ipady=3)
            self.count_labels[cls] = badge

    # ── run / stop ────────────────────────────────────────────────
    def _toggle_run(self):
        if not self.running:
            if not self.class_names:
                messagebox.showwarning("No model", "Please browse and select a model first.")
                return
            self.running = True
            self.run_btn.config(text="■  STOP", bg=RED, activebackground="#b71c1c")
            self.canvas.itemconfig(self.status_dot, fill=GREEN)
            self._start_detection_loop()
        else:
            self._stop()

    def _stop(self):
        self.running = False
        self.run_btn.config(text="▶  RUN", bg=ACCENT, activebackground=ACCENT_DARK)
        self.fps_label.config(text="---.--", fg=YELLOW)
        self.fps_bar.place(width=0)
        self.canvas.itemconfig(self.status_dot, fill="#333333")
        self.canvas.itemconfig(self.cv_label,
                                text="Video  RealTime", fill="#3a5f8a")
        self.canvas.config(bg=CANVAS_BG)
        for lbl in self.count_labels.values():
            lbl.config(text="0", fg=RED)

    # ── fake inference loop (replace body with real model later) ──
    def _start_detection_loop(self):
        def loop():
            while self.running:
                t0 = time.time()
                counts = {
                    cls: random.randint(0, 5) if random.random() > 0.35 else 0
                    for cls in self.class_names
                }
                self.root.after(0, self._update_ui, counts, self.fps_val)
                time.sleep(0.08)
                self.fps_val = 1.0 / (time.time() - t0 + 1e-9)

        threading.Thread(target=loop, daemon=True).start()

    # ── update UI (main thread) ───────────────────────────────────
    def _update_ui(self, counts: dict, fps: float):
        # counts per class
        for cls, lbl in self.count_labels.items():
            n = counts.get(cls, 0)
            lbl.config(text=str(n), fg=GREEN if n > 0 else RED)

        # FPS gauge
        fps_color = GREEN if fps >= 25 else (YELLOW if fps >= 10 else RED)
        self.fps_label.config(text=f"{fps:.2f}", fg=fps_color)
        bar_w = int(min(fps, 60) / 60 * 80)
        self.fps_bar.place(width=min(bar_w, 80))
        self.fps_bar.config(bg=fps_color)

        # canvas pulse
        v = random.randint(14, 26)
        self.canvas.config(bg=f"#{v:02x}{v+4:02x}{v+9:02x}")

        detected = [f"{cls} × {n}" for cls, n in counts.items() if n > 0]
        self.canvas.itemconfig(
            self.cv_label,
            text="\n\n".join(detected) if detected else "[ scanning... ]",
            fill=GREEN if detected else "#3a7a5f")


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeDetectionApp(root)
    root.mainloop()