"""
YOLOv8 Real-Time Detection GUI
Requires: opencv-python, onnxruntime, Pillow
  pip install opencv-python onnxruntime Pillow

Logic:
- ตีกรอบ bounding box บนภาพเมื่อตรวจพบ object (confidence >= CONF_THRESHOLD)
- Class จะถูกนับเมื่อ confidence >= 80% ติดต่อกัน 2 วินาที (STABLE_SEC)
- ถ้า object หายไป จำนวนจะลดลงทันที
- FPS อัปเดตทุก 1 วินาที
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading, time, os, ast
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# ── palette ──────────────────────────────────────────────────────
BG, PANEL, CAM  = "#1e2130", "#252a3a", "#0d1117"
ACCENT, ADARK   = "#2979ff", "#1a56cc"
RED, GREEN, YEL = "#f44336", "#00e676", "#ffd740"
TMAIN, TDIM, BD = "#e8eaf6", "#7986cb", "#3d4466"
CAM_W, CAM_H    = 340, 260
ICONS           = ["●","◆","▲","★","■","✦","◉","▶"]

CONF_THRESHOLD  = 0.80   # ขั้นต่ำ confidence ที่จะนับและวาดกรอบ
STABLE_SEC      = 2.0    # ต้องเจอติดต่อกันกี่วินาทีจึงนับ
FPS_UPDATE_SEC  = 1.0    # อัปเดต FPS ทุกกี่วินาที

# สีกรอบแต่ละ class (BGR สำหรับ cv2, RGB สำหรับแสดงผล)
BOX_COLORS_RGB = [
    (0, 230, 118), (41, 121, 255), (255, 215, 0),
    (255, 87,  34), (156, 39, 176), (0, 188, 212),
    (233, 30,  99), (76, 175, 80),
]


def read_class_names(path: str) -> list[str]:
    """Extract class names from YOLOv8 .onnx custom_metadata_map."""
    if not HAS_ORT:
        return []
    try:
        meta = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        ).get_modelmeta().custom_metadata_map
        raw = meta.get("names", "")
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return [parsed[k] for k in sorted(parsed)]
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLOv8 · Real-Time Detection")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.running       = False
        self.fps_val       = 0.0
        self._fps_display  = 0.0          # ค่าที่แสดงบน UI (อัปเดตทุก 1 วิ)
        self._fps_last_upd = 0.0          # timestamp ที่อัปเดต FPS ล่าสุด
        self.class_names   : list[str]          = []
        self.count_labels  : dict[str,tk.Label] = {}
        self.cap           : cv2.VideoCapture | None = None
        self.session       = None
        self.input_name    = ""
        self._photo        = None

        # ── stability tracking ──────────────────────────────────
        # first_seen[cls] = timestamp ที่เริ่มเห็น class นี้ครั้งแรก (ในรอบล่าสุด)
        # None หมายความว่าหายไปแล้ว
        self._first_seen   : dict[str, float | None] = {}
        # confirmed_count[cls] = จำนวนที่ผ่าน 2 วิแล้ว (แสดงบน UI)
        self._confirmed    : dict[str, int] = {}

        self._build_ui()

    # ── helpers ──────────────────────────────────────────────────
    def _card(self, p, **kw) -> tk.Frame:
        return tk.Frame(p, bg=PANEL, highlightthickness=1,
                        highlightbackground=BD, **kw)

    def _btn(self, p, text, cmd, bg, fg="white", **kw) -> tk.Button:
        kw.setdefault("activebackground", bg)
        kw.setdefault("activeforeground", fg)
        return tk.Button(p, text=text, command=cmd, bg=bg, fg=fg,
                         relief="flat", cursor="hand2", **kw)

    # ── build UI ─────────────────────────────────────────────────
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG, padx=14, pady=12)
        outer.pack()

        # camera canvas
        wrap = tk.Frame(outer, bg=CAM, highlightthickness=1, highlightbackground=BD)
        wrap.grid(row=0, column=0, padx=(0,14), sticky="n")
        self.canvas = tk.Canvas(wrap, width=CAM_W, height=CAM_H,
                                bg=CAM, highlightthickness=0)
        self.canvas.pack(padx=2, pady=2)
        for cx,cy,dx,dy in [(8,8,1,1),(CAM_W-8,8,-1,1),
                             (8,CAM_H-8,1,-1),(CAM_W-8,CAM_H-8,-1,-1)]:
            self.canvas.create_line(cx,cy,cx+dx*18,cy, fill=ACCENT, width=2)
            self.canvas.create_line(cx,cy,cx,cy+dy*18, fill=ACCENT, width=2)
        self.cv_text = self.canvas.create_text(
            CAM_W//2, CAM_H//2, text="No Camera",
            fill="#3a5f8a", font=("Consolas",15,"bold"))
        self.status  = self.canvas.create_oval(10,8,22,20, fill="#444", outline="")

        # right panel
        right = tk.Frame(outer, bg=BG)
        right.grid(row=0, column=1, sticky="n")

        hdr = self._card(right)
        hdr.pack(fill="x", pady=(0,10))
        tk.Label(hdr, text="  DETECTION  PANEL",
                 bg=PANEL, fg=TDIM, font=("Consolas",9,"bold"), pady=5).pack(side="left")

        # model selector
        mc = self._card(right)
        mc.pack(fill="x", pady=(0,10), ipadx=8, ipady=4)
        tk.Label(mc, text="MODEL", bg=PANEL, fg=TDIM,
                 font=("Consolas",8,"bold")).pack(anchor="w", padx=8, pady=(6,2))
        mrow = tk.Frame(mc, bg=PANEL); mrow.pack(fill="x", padx=8, pady=(0,6))
        self.model_var   = tk.StringVar()
        self.model_entry = tk.Entry(
            mrow, textvariable=self.model_var, width=15,
            bg="#0d1117", fg=TDIM, insertbackground=TMAIN, relief="flat",
            font=("Consolas",10), highlightthickness=1,
            highlightbackground=BD, highlightcolor=ACCENT)
        self.model_entry.pack(side="left", ipady=4)
        self.model_entry.insert(0, "no model loaded")
        self._btn(mrow, "browse", self._browse, bg="#2e3450", fg=TMAIN,
                  font=("Consolas",9), padx=8, pady=4,
                  highlightthickness=1, highlightbackground=BD
                  ).pack(side="left", padx=(6,0))

        # RUN button
        self.run_btn = self._btn(right, "▶  RUN", self._toggle,
                                  bg=ACCENT, font=("Consolas",14,"bold"),
                                  width=14, pady=8)
        self.run_btn.pack(pady=(0,10))

        # object count card
        self.obj_card = self._card(right)
        self.obj_card.pack(fill="x", pady=(0,10), ipadx=8, ipady=4)
        tk.Label(self.obj_card, text="OBJECT  COUNT",
                 bg=PANEL, fg=TDIM, font=("Consolas",8,"bold")
                 ).grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(6,4))
        tk.Frame(self.obj_card, bg=BD, height=1
                 ).grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0,4))
        self.obj_ph = tk.Label(self.obj_card,
                                text="  — browse a model to see classes —",
                                bg=PANEL, fg=TDIM, font=("Consolas",9), pady=6)
        self.obj_ph.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0,4))

        # FPS
        fc = self._card(right); fc.pack(fill="x", pady=(0,10), ipadx=8, ipady=4)
        fi = tk.Frame(fc, bg=PANEL); fi.pack(fill="x", padx=8, pady=4)
        tk.Label(fi, text="FPS", bg=PANEL, fg=TDIM,
                 font=("Consolas",9,"bold")).pack(side="left")
        self.fps_lbl = tk.Label(fi, text="---.--", bg=PANEL, fg=YEL,
                                 font=("Consolas",13,"bold"))
        self.fps_lbl.pack(side="left", padx=10)
        bbg = tk.Frame(fi, bg=BD, width=80, height=8)
        bbg.pack(side="left"); bbg.pack_propagate(False)
        self.fps_bar = tk.Frame(bbg, bg=YEL, height=8)
        self.fps_bar.place(x=0, y=0, height=8, width=0)

        # EXIT
        self._btn(right, "EXIT", self._quit, bg=RED,
                  font=("Consolas",11,"bold"), width=8, pady=5,
                  activebackground="#b71c1c").pack(anchor="e", pady=(4,0))

    # ── load ONNX session ────────────────────────────────────────
    def _load_session(self, path: str):
        """โหลด ONNX session เก็บไว้ใช้ใน _loop()"""
        if not HAS_ORT:
            return
        self.session    = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # ── parse YOLOv8 raw output ───────────────────────────────────
    @staticmethod
    def _parse_yolo(raw: np.ndarray, orig_w: int, orig_h: int,
                    class_names: list, conf_thr: float = 0.25,
                    iou_thr: float = 0.45):
        """
        raw shape: [1, 4+num_classes, num_anchors]  (YOLOv8 ONNX default)
        คืนค่า list of (class_name, confidence, x1, y1, x2, y2)
        โดย x1..y2 อยู่ใน orig_w x orig_h space
        """
        preds = raw[0].T                        # → [num_anchors, 4+nc]
        boxes_xywh = preds[:, :4]              # cx,cy,w,h  (normalized 0-1 ใน 320x320)
        scores     = preds[:, 4:]              # [num_anchors, nc]

        class_ids  = scores.argmax(axis=1)
        confs      = scores.max(axis=1)

        mask = confs >= conf_thr
        boxes_xywh = boxes_xywh[mask]
        confs      = confs[mask]
        class_ids  = class_ids[mask]

        if len(confs) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2  (scale ถึง orig size)
        cx, cy, w, h = boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,2], boxes_xywh[:,3]
        x1 = ((cx - w/2) * orig_w).astype(int)
        y1 = ((cy - h/2) * orig_h).astype(int)
        x2 = ((cx + w/2) * orig_w).astype(int)
        y2 = ((cy + h/2) * orig_h).astype(int)

        # NMS per-class — ข้าม cid ที่ไม่อยู่ใน class_names
        results = []
        boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1).astype(float)
        for cid in np.unique(class_ids):
            if cid >= len(class_names):   # class id เกิน → ข้าม
                continue
            idx  = np.where(class_ids == cid)[0]
            keep = App._nms(boxes_xyxy[idx], confs[idx], iou_thr)
            for k in keep:
                i = idx[k]
                # คืน (cls_name, cid, conf, x1, y1, x2, y2)
                results.append((class_names[cid], int(cid),
                                float(confs[i]),
                                int(x1[i]), int(y1[i]),
                                int(x2[i]), int(y2[i])))
        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
        """Greedy NMS — คืน index ที่เหลือ"""
        order = scores.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
            yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
            xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
            yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
            area_o = (boxes[order[1:],2]-boxes[order[1:],0]) * (boxes[order[1:],3]-boxes[order[1:],1])
            iou   = inter / (area_i + area_o - inter + 1e-9)
            order = order[1:][iou < iou_thr]
        return keep

    # ── browse ───────────────────────────────────────────────────
    def _browse(self):
        if self.running: return
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        path = filedialog.askopenfilename(
            initialdir=models_dir, title="Select ONNX model",
            filetypes=[("ONNX files","*.onnx"),("All files","*.*")])
        if not path: return
        self.model_var.set(os.path.basename(path))
        self.model_entry.config(fg=TMAIN)
        def _bg():
            names = read_class_names(path)
            self._load_session(path)
            self.root.after(0, self._load_classes, names)
        threading.Thread(target=_bg, daemon=True).start()

    def _load_classes(self, names: list[str]):
        for w in list(self.obj_card.grid_slaves()):
            if int(w.grid_info().get("row", 0)) >= 2:
                w.destroy()
        self.count_labels.clear()
        self.class_names = names
        self._first_seen  = {cls: None for cls in names}
        self._confirmed   = {cls: 0    for cls in names}

        if not names:
            self.obj_ph.config(text="  ⚠  no class names found in metadata")
            self.obj_ph.grid(row=2, column=0, columnspan=3,
                              sticky="w", padx=8, pady=(0,4))
            return

        for i, cls in enumerate(names):
            r = i + 2
            tk.Label(self.obj_card, text=f"  {ICONS[i%len(ICONS)]}  {cls}",
                     bg=PANEL, fg=GREEN, font=("Consolas",11)
                     ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
            tk.Label(self.obj_card, text="=", bg=PANEL, fg=TDIM,
                     font=("Consolas",11)).grid(row=r, column=1, padx=8)
            badge = tk.Label(self.obj_card, text="0", bg="#0d1117", fg=RED,
                              font=("Consolas",13,"bold"), width=4,
                              highlightthickness=1, highlightbackground=BD)
            badge.grid(row=r, column=2, padx=(0,10), pady=3, ipady=3)
            self.count_labels[cls] = badge

    # ── run / stop ───────────────────────────────────────────────
    def _toggle(self):
        if self.running:
            self._stop(); return
        if not self.class_names:
            messagebox.showwarning("No model", "Please select a model first.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Cannot open camera (index 0).")
            return
        self.running = True
        self._fps_last_upd = time.time()
        self._first_seen   = {cls: None for cls in self.class_names}
        self._confirmed    = {cls: 0    for cls in self.class_names}
        self.run_btn.config(text="■  STOP", bg=RED, activebackground="#b71c1c")
        self.canvas.itemconfig(self.status, fill=GREEN)
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        self.run_btn.config(text="▶  RUN", bg=ACCENT, activebackground=ADARK)
        self.fps_lbl.config(text="---.--", fg=YEL)
        self.fps_bar.place(width=0)
        self.canvas.itemconfig(self.status, fill="#444")
        self.canvas.delete("frame"); self._photo = None
        self.canvas.itemconfig(self.cv_text, state="normal", text="No Camera")
        for lbl in self.count_labels.values():
            lbl.config(text="0", fg=RED)

    # ── inference / camera loop ───────────────────────────────────
    def _loop(self):
        frame_count = 0
        fps_acc     = 0.0

        while self.running:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok: break

            INF_SIZE       = 320
            orig_h, orig_w = frame.shape[:2]

            # ── Preprocess: BGR→RGB → float32/255 → CHW → [1,3,320,320] ──
            blob = cv2.resize(frame, (INF_SIZE, INF_SIZE))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            blob = blob.astype(np.float32) / 255.0          # normalize 0-1
            blob = blob.transpose(2, 0, 1)                  # HWC → CHW
            blob = np.expand_dims(blob, axis=0)             # → [1,3,320,320]

            # ── REAL INFERENCE ───────────────────────────────────────────
            if self.session:
                raw        = self.session.run(None, {self.input_name: blob})[0]
                detections = self._parse_yolo(raw, orig_w, orig_h,
                                              self.class_names, CONF_THRESHOLD)
            else:
                detections = []
            # ─────────────────────────────────────────────────────────────

            # scale bbox จาก orig size → display size (CAM_W x CAM_H)
            sx, sy = CAM_W / orig_w, CAM_H / orig_h
            detections = [
                (cls, cid, conf,
                 int(x1*sx), int(y1*sy),
                 int(x2*sx), int(y2*sy))
                for cls, cid, conf, x1, y1, x2, y2 in detections
            ]

            now = time.time()

            # กรอง confidence < threshold  (tuple: cls, cid, conf, x1,y1,x2,y2)
            valid = [(cls,cid,conf,x1,y1,x2,y2)
                     for cls,cid,conf,x1,y1,x2,y2 in detections
                     if conf >= CONF_THRESHOLD]

            # อัปเดต stability tracker
            detected_classes = {}  # cls -> count ในเฟรมนี้
            for cls,cid,conf,*_ in valid:
                detected_classes[cls] = detected_classes.get(cls, 0) + 1

            counts = {}
            for cls in self.class_names:
                if cls in detected_classes:
                    if self._first_seen[cls] is None:
                        self._first_seen[cls] = now
                    if (now - self._first_seen[cls]) >= STABLE_SEC:
                        self._confirmed[cls] = detected_classes[cls]
                    counts[cls] = self._confirmed[cls]
                else:
                    self._first_seen[cls] = None
                    self._confirmed[cls]  = 0
                    counts[cls]           = 0

            # วาด bounding box บน frame ขนาด display
            frame_draw = cv2.resize(frame, (CAM_W, CAM_H))
            for cls,cid,conf,x1,y1,x2,y2 in valid:
                r,g,b = BOX_COLORS_RGB[cid % len(BOX_COLORS_RGB)]
                color_bgr = (b, g, r)
                cv2.rectangle(frame_draw, (x1,y1), (x2,y2), color_bgr, 2)
                label = f"{cls} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # พื้นหลัง label
                cv2.rectangle(frame_draw,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              color_bgr, -1)
                cv2.putText(frame_draw, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

            # FPS: นับสะสม อัปเดตทุก FPS_UPDATE_SEC
            elapsed_frame = time.time() - t0
            frame_count  += 1
            fps_acc       += 1.0 / (elapsed_frame + 1e-9)
            if (now - self._fps_last_upd) >= FPS_UPDATE_SEC:
                self._fps_display  = fps_acc / frame_count
                fps_acc, frame_count = 0.0, 0
                self._fps_last_upd = now

            self.root.after(0, self._update, img, counts, self._fps_display)

    # ── update UI ────────────────────────────────────────────────
    def _update(self, img, counts: dict, fps: float):
        self._photo = img
        self.canvas.delete("frame")
        self.canvas.create_image(0, 0, anchor="nw", image=img, tags="frame")
        self.canvas.tag_lower("frame")
        self.canvas.itemconfig(self.cv_text, state="hidden")

        for cls, lbl in self.count_labels.items():
            n = counts.get(cls, 0)
            lbl.config(text=str(n), fg=GREEN if n > 0 else RED)

        col = GREEN if fps >= 25 else (YEL if fps >= 10 else RED)
        self.fps_lbl.config(text=f"{fps:.2f}", fg=col)
        self.fps_bar.place(width=int(min(fps,60)/60*80))
        self.fps_bar.config(bg=col)

    # ── quit ─────────────────────────────────────────────────────
    def _quit(self):
        self.running = False
        if self.cap: self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()