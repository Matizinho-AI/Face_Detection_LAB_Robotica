#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, shutil, cv2, torch, numpy as np, pandas as pd, tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# — Rutas y carpetas —
DATASET_DIR   = "dataset_ai"
UNKNOWN_DIR   = "unknowns"
ATT_CSV       = "attendance_ai.csv"
PROFILES_FILE = "profiles.json"
for d in (DATASET_DIR, UNKNOWN_DIR):
    os.makedirs(d, exist_ok=True)

# — Parámetros —
TH_PROFILE     = 0.85
TH_KEEP        = 0.70
MIN_SAMPLES    = 10
MAX_CAMERAS    = 10

# — Helpers de persistencia —
def load_profiles():
    return json.load(open(PROFILES_FILE)) if os.path.isfile(PROFILES_FILE) else {}

def save_profiles(p):
    json.dump(p, open(PROFILES_FILE, "w"), indent=2)

def load_att():
    if os.path.isfile(ATT_CSV):
        df = pd.read_csv(ATT_CSV, index_col="ID", dtype=str)
    else:
        df = pd.DataFrame(columns=["Nombre","Apellido"])
        df.index.name = "ID"
        df.to_csv(ATT_CSV)
    for c in ("Nombre","Apellido"):
        if c not in df.columns:
            df.insert(0, c, "")
    return df

def save_att(df):
    df.to_csv(ATT_CSV)

# — Motor de IA y asistencia —
class FaceAttendance:
    def __init__(self, cam_idx, log_fn):
        self.log = log_fn
        self.profiles   = load_profiles()
        self.cand       = {}
        self.assigned   = {}
        self.last_bin   = {}
        self.last_photo = {}
        self.last_un    = {}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.log(f"Dispositivo: {dev.upper()}")
        self.mtcnn  = MTCNN(keep_all=True, device=dev)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(dev)
        self.cap    = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("No se abrió la cámara")
        self.att_df = load_att()

    def process(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        h, w = frame.shape[:2]
        boxes, _ = self.mtcnn.detect(frame)
        now = datetime.now()
        block = f"{now.date()}_{now.hour:02d}-{(now.hour+1)%24:02d}"
        keys = set()

        if boxes is not None:
            aligned = self.mtcnn(frame)
            embs    = self.resnet(aligned).detach().cpu().numpy()

            for emb, box in zip(embs, boxes):
                x1,y1,x2,y2 = map(int, box)
                # clamp
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                key = (x1//20, y1//20, (x2-x1)//20, (y2-y1)//20)
                keys.add(key)
                cd = self.cand.setdefault(key, {"embs":[], "boxes":[], "count":0})
                cd["embs"].append(emb)
                cd["boxes"].append((x1,y1,x2,y2))
                cd["count"] += 1

                # match perfil existente
                best_sim, best_id = -1, None
                for pid, p in self.profiles.items():
                    cent = np.array(p["centroid"])
                    sim = np.dot(cent,emb)/(np.linalg.norm(cent)*np.linalg.norm(emb))
                    if sim > best_sim:
                        best_sim, best_id = sim, pid

                name = None
                if best_sim >= TH_PROFILE:
                    name = best_id
                    cnt  = self.profiles[name]["count"]
                    newc = ((cnt*np.array(self.profiles[name]["centroid"])) + emb)/(cnt+1)
                    self.profiles[name]["centroid"] = newc.tolist()
                    self.profiles[name]["count"]    = cnt+1
                    # foto diaria de re-entreno
                    if self.last_photo.get(name) != str(now.date()):
                        roi = frame[y1:y2, x1:x2]
                        if roi.size:
                            fld = os.path.join(DATASET_DIR, name)
                            os.makedirs(fld, exist_ok=True)
                            fn = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
                            cv2.imwrite(os.path.join(fld, fn), roi)
                            self.last_photo[name] = str(now.date())

                elif key in self.assigned and best_sim >= TH_KEEP:
                    name = self.assigned[key]

                # enrolar nuevo perfil
                elif cd["count"] >= MIN_SAMPLES:
                    arr  = np.stack(cd["embs"])
                    sims = (arr@arr.T)/(np.linalg.norm(arr,axis=1)[:,None]*np.linalg.norm(arr,axis=1)[None,:])
                    if sims.min() > TH_PROFILE:
                        name = f"person_{len(self.profiles)}"
                        cent = arr.mean(axis=0)
                        self.profiles[name] = {
                            "centroid": cent.tolist(),
                            "count": cd["count"],
                            "Nombre": "",
                            "Apellido": ""
                        }
                        fld = os.path.join(DATASET_DIR, name)
                        os.makedirs(fld, exist_ok=True)
                        for i,(bx,by,bx2,by2) in enumerate(cd["boxes"]):
                            roi = frame[by:by2, bx:bx2]
                            if roi.size:
                                fn = f"{now:%Y%m%d_%H%M%S}_{i}.jpg"
                                cv2.imwrite(os.path.join(fld, fn), roi)
                        self.log(f"Nuevo perfil: {name}")
                    del self.cand[key]

                # marcar asistencia 1h
                if name:
                    if self.last_bin.get(name) != block:
                        self.att_df[block] = 0
                        if name not in self.att_df.index:
                            self.att_df.loc[name] = ["",""] + [0]*(self.att_df.shape[1]-2)
                        self.att_df.at[name, block] = 1
                        self.last_bin[name] = block
                        save_att(self.att_df)
                        save_profiles(self.profiles)
                    self.assigned[key] = name
                else:
                    # unknown diario
                    if self.last_un.get(key) != str(now.date()):
                        roi = frame[y1:y2, x1:x2]
                        if roi.size:
                            fn = f"{now:%Y%m%d_%H%M%S}_{key[0]}_{key[1]}.jpg"
                            cv2.imwrite(os.path.join(UNKNOWN_DIR, fn), roi)
                            self.last_un[key] = str(now.date())
                    self.assigned.pop(key, None)

                # dibujo bounding box usando datos de self.profiles
                col = (0,255,0) if name else (0,0,255)
                if name:
                    prof = self.profiles.get(name, {})
                    nom  = prof.get("Nombre", "")
                    ape  = prof.get("Apellido", "")
                    label = f"{nom} {ape}".strip() if (nom or ape) else name
                else:
                    label = "Unknown"
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # limpiar buffers obsoletos
        self.cand     = {k:v for k,v in self.cand.items()     if k in keys}
        self.assigned = {k:v for k,v in self.assigned.items() if k in keys}

        # devolver en RGB para la GUI
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()

# — GUI principal —
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Face Attendance")
        self.geometry("1050x740")
        self.det = None; self.upd = None

        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True)
        self.tab_det   = ttk.Frame(nb); nb.add(self.tab_det,   text="Detección")
        self.tab_prof  = ttk.Frame(nb); nb.add(self.tab_prof,  text="Perfiles")
        self.tab_merge = ttk.Frame(nb); nb.add(self.tab_merge, text="Fusionar perfiles")

        self.build_det()
        self.build_merge()
        self.build_prof()

    # DETECCIÓN
    def build_det(self):
        f = self.tab_det
        cams = [i for i in range(MAX_CAMERAS) if cv2.VideoCapture(i).isOpened()]
        ttk.Label(f, text="Cámara:").pack(anchor="w", padx=10, pady=5)
        self.cam_sel = ttk.Combobox(f, values=cams, state="readonly")
        if cams: self.cam_sel.current(0)
        self.cam_sel.pack(anchor="w", padx=10)

        bf = ttk.Frame(f); bf.pack(pady=5)
        ttk.Button(bf, text="Iniciar", command=self.start).pack(side="left", padx=5)
        self.bt_stop = ttk.Button(bf, text="Detener", command=self.stop, state="disabled")
        self.bt_stop.pack(side="left")

        self.lbl = ttk.Label(f); self.lbl.pack(padx=10, pady=10)
        self.log = scrolledtext.ScrolledText(f, height=6, state="disabled")
        self.log.pack(fill="x", padx=10, pady=(0,10))

    # FUSIONAR PERFILES
    def build_merge(self):
        f = self.tab_merge
        ttk.Label(f, text="Fusionar dos perfiles distintos").pack(pady=5)
        self.merge_a = ttk.Combobox(f, values=[])
        self.merge_b = ttk.Combobox(f, values=[])
        self.merge_a.pack(pady=5); self.merge_b.pack(pady=5)
        ttk.Button(f, text="Fusionar", command=self.merge_profiles).pack(pady=10)
        self.refresh_merge_lists()

    # PERFILES CRUD
    def build_prof(self):
        f = self.tab_prof
        left = ttk.Frame(f); left.pack(side="left", fill="y", padx=10, pady=10)
        ttk.Label(left, text="Perfiles:").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("Nombre","Apellido"), show="headings", height=25)
        self.tree.heading("Nombre", text="Nombre"); self.tree.heading("Apellido", text="Apellido")
        self.tree.pack()
        ttk.Button(left, text="Recargar", command=self.load_prof).pack(pady=5)

        right = ttk.Frame(f); right.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.foto = ttk.Label(right); self.foto.pack(pady=10)
        frm = ttk.Frame(right); frm.pack()
        self.nv = tk.StringVar(); self.sv = tk.StringVar()
        ttk.Label(frm, text="Nombre:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.nv).grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="Apellido:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.sv).grid(row=1, column=1, sticky="w")

        bf = ttk.Frame(right); bf.pack(pady=10)
        ttk.Button(bf, text="Guardar",  command=self.save_prof).pack(side="left", padx=5)
        ttk.Button(bf, text="Eliminar", command=self.del_prof).pack(side="left")

        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.load_prof()

    # helpers GUI
    def logmsg(self, m):
        self.log.config(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S} – {m}\n")
        self.log.see("end"); self.log.config(state="disabled")

    # captura loop
    def start(self):
        idx = int(self.cam_sel.get())
        try:
            self.det = FaceAttendance(idx, self.logmsg)
        except Exception as e:
            self.logmsg(f"ERROR: {e}"); return
        self.bt_stop.config(state="normal"); self.cam_sel.config(state="disabled")
        self._upd()

    def _upd(self):
        frame = self.det.process()
        if frame is None:
            return self.stop()
        img = ImageTk.PhotoImage(Image.fromarray(frame).resize((640,360)))
        self.lbl.img = img; self.lbl.config(image=img)
        self.upd = self.after(30, self._upd)

    def stop(self):
        if self.upd: self.after_cancel(self.upd)
        if self.det: self.det.release(); self.det = None
        self.bt_stop.config(state="disabled"); self.cam_sel.config(state="readonly")
        self.logmsg("Detenido")

    # perfiles CRUD
    def load_prof(self):
        self.df = load_att()
        self.tree.delete(*self.tree.get_children())
        for pid,row in self.df.iterrows():
            self.tree.insert("", "end", iid=pid, values=(row["Nombre"],row["Apellido"]))
        self.refresh_merge_lists()

    def on_select(self, _=None):
        sel = self.tree.selection()
        if not sel: return
        pid = sel[0]
        self.nv.set(self.df.at[pid,"Nombre"])
        self.sv.set(self.df.at[pid,"Apellido"])
        fld = os.path.join(DATASET_DIR, pid)
        if os.path.isdir(fld):
            imgs = [f for f in os.listdir(fld) if f.endswith(".jpg")]
            if imgs:
                im = Image.open(os.path.join(fld,imgs[0])).resize((180,180))
                tkim = ImageTk.PhotoImage(im)
                self.foto.img = tkim; self.foto.config(image=tkim)

    def save_prof(self):
        sel = self.tree.selection()
        if not sel: return
        pid = sel[0]
        # actualizar attendance df
        self.df.at[pid,"Nombre"]   = self.nv.get()
        self.df.at[pid,"Apellido"] = self.sv.get()
        save_att(self.df)
        # actualizar perfiles IA
        if self.det and pid in self.det.profiles:
            self.det.profiles[pid]["Nombre"] = self.nv.get()
            self.det.profiles[pid]["Apellido"] = self.sv.get()
        self.tree.item(pid, values=(self.nv.get(),self.sv.get()))
        self.logmsg(f"Perfil {pid} guardado")
        self.refresh_merge_lists()

    def del_prof(self):
        sel = self.tree.selection()
        if not sel: return
        pid = sel[0]
        if not messagebox.askyesno("Eliminar", f"¿Eliminar perfil {pid}?"):
            return
        shutil.rmtree(os.path.join(DATASET_DIR,pid), ignore_errors=True)
        # actualizar perfiles IA
        if self.det and pid in self.det.profiles:
            self.det.profiles.pop(pid, None)
            self.det.att_df.drop(index=pid, inplace=True)
        profs = load_profiles(); profs.pop(pid,None); save_profiles(profs)
        self.df.drop(index=pid, inplace=True); save_att(self.df)
        self.tree.delete(pid)
        self.logmsg(f"Perfil {pid} eliminado")
        self.refresh_merge_lists()

    # fusionar perfiles
    def refresh_merge_lists(self):
        ids = list(load_profiles().keys())
        if hasattr(self, "merge_a") and hasattr(self, "merge_b"):
            self.merge_a["values"] = ids
            self.merge_b["values"] = ids

    def merge_profiles(self):
        a = self.merge_a.get(); b = self.merge_b.get()
        if not a or not b or a == b:
            messagebox.showinfo("Fusionar","Elige dos perfiles distintos"); return
        profs = load_profiles()
        ca, cb = profs[a], profs[b]
        total = ca["count"] + cb["count"]
        newc = ((ca["count"]*np.array(ca["centroid"]) + cb["count"]*np.array(cb["centroid"])) / total).tolist()
        ca["centroid"], ca["count"] = newc, total
        fa, fb = os.path.join(DATASET_DIR,a), os.path.join(DATASET_DIR,b)
        if os.path.isdir(fb):
            for f in os.listdir(fb):
                shutil.move(os.path.join(fb,f), fa)
            shutil.rmtree(fb, ignore_errors=True)
        df = load_att()
        if b in df.index:
            if a not in df.index:
                df.loc[a] = df.loc[b]
            else:
                for col in df.columns[2:]:
                    df.at[a,col] = str(int(df.at[a,col] or 0) | int(df.at[b,col] or 0))
            df.drop(index=b, inplace=True)
            save_att(df)
        profs.pop(b); save_profiles(profs)
        # actualizar IA
        if self.det:
            if b in self.det.profiles: self.det.profiles.pop(b)
            if a in self.det.profiles:
                self.det.profiles[a]["centroid"] = newc
                self.det.profiles[a]["count"] = total
        self.logmsg(f"Fusionados {a} ← {b}")
        self.load_prof()

    def on_close(self):
        self.stop()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
