# -*- coding: utf-8 -*-
"""
tomato_pipeline.py  (sürüm 3 — kamera görüş açısı tasarımı)
============================================================
Çalışma: Çok açılı görüntülerde kamera görüş açısının domates büyüme evresi
sınıflandırmasına etkisi — YOLO11n-cls ve YOLO26n-cls.

Tasarım
  * 50 BBCH kodu -> 6 ana evre (kodun ilk hanesi).
  * Her bitki-gün oturumundan, dört kamerada da bulunan pozlar arasından
    sabit tohumlu ve deterministik olarak TEK poz seçilir; o pozun 4 kamera görüntüsü alınır.
  * Bitkiler StratifiedGroupKFold(n_splits=5) ile BİR KEZ beş kata ayrılır
    (grup = bitki, hedef = ana evre). Aynı kat manifesti tüm kamera ve modellerde kullanılır.
  * Dış kat k: test = k, doğrulama = (k+1) mod 5, eğitim = kalan 3 kat.
  * 2 model x 4 kamera x 5 kat = 40 eğitim, tek sabit eğitim tohumu.
  * Kat-dışı (OOF) tahminler birleştirilir; dört kameranın olasılıkları ortalanarak geç füzyon.
  * Bitki kümeli, eşleştirilmiş bootstrap; aile içi Holm düzeltmesi.

Aşamalar (her biri kaldığı yerden devam eder; kalıcı çıktılar Google Drive'dadır)
  1 stage_index       samples.json -> metadata_full.csv + veri denetimi
  2 stage_select      poz seçimi -> selection_manifest.csv, subset_images.csv
  3 stage_images      seçmeli indirme (gerekirse tam indirme) -> ön işleme -> tar parçaları
  4 stage_folds       bitki düzeyinde 5 kat -> plant_folds.csv
  5 stage_train_eval  40 eğitim + test-katı tahminleri (resume destekli)
  6 stage_oof         OOF birleştirme + geç füzyon
  7 stage_efficiency  parametre, GFLOPs, gecikme: tek kamera vs dört kamera
  8 stage_analysis    tablolar (CSV + XLSX), şekiller (PNG 300 dpi + PDF), özet JSON

Kaynak veri: Zhang et al. (2026) Scientific Data 13:309 (CC BY 4.0). HF aynası: Voxel51/tomato-map.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------
STAGE_TO_CLASS = {1: "0_S1", 2: "1_S2", 5: "2_S5", 6: "3_S6", 7: "4_S7", 8: "5_S8"}
CLASSES = list(STAGE_TO_CLASS.values())
K = len(CLASSES)
CLASS_LABELS_TR = {
    "0_S1": "Yaprak gelişimi (1)",
    "1_S2": "Yan sürgün oluşumu (2)",
    "2_S5": "Çiçek salkımı çıkışı (5)",
    "3_S6": "Çiçeklenme (6)",
    "4_S7": "Meyve gelişimi (7)",
    "5_S8": "Olgunlaşma (8)",
}
CLASS_SHORT = {"0_S1": "S1", "1_S2": "S2", "2_S5": "S5", "3_S6": "S6", "4_S7": "S7", "5_S8": "S8"}
FNAME_RE = re.compile(r"pi(\d)_(\d+)_(\d+)_(\d+)_(\d{14})")
FUSION = "fusion"


@dataclass
class Config:
    drive_root: str = "/content/drive/MyDrive/YOLODomatesEylul2026"
    local_root: str = "/content/work"
    hf_repo: str = "Voxel51/tomato-map"
    cameras: tuple = (1, 2, 3, 4)
    # Kamera numarası -> makalede kullanılacak etiket. Derece bilgisi kaynaktan
    # doğrulanmadan buraya açı yazmayın.
    camera_labels: dict = field(default_factory=lambda: {1: "K1", 2: "K2", 3: "K3", 4: "K4"})
    pose_seed: int = 2026
    fold_seed: int = 2026
    n_folds: int = 5
    train_seed: int = 0
    models: tuple = ("yolo11n-cls.pt", "yolo26n-cls.pt")
    prep_size: int = 384
    imgsz: int = 320
    epochs: int = 50
    patience: int = 15
    batch: int = 64
    workers: int = 8
    shard_size: int = 1000
    download_workers: int = 16
    allow_full_download: bool = True     # seçmeli indirme olmazsa tam indirmeye geç
    bootstrap_B: int = 2000
    bootstrap_seed: int = 12345
    latency_runs: int = 100
    offline_src: str | None = None       # yalnızca yerel test için

    @property
    def D(self) -> Path:
        return Path(self.drive_root)

    @property
    def L(self) -> Path:
        return Path(self.local_root)

    def dirs(self):
        return {
            "cache": self.D / "cache",
            "shards": self.D / "cache" / "shards",
            "manifests": self.D / "manifests",
            "runs": self.D / "runs",
            "oof": self.D / "results" / "oof",
            "tables": self.D / "results" / "tables",
            "figures": self.D / "results" / "figures",
            "logs": self.D / "logs",
            "raw": self.L / "raw",
            "images": self.L / "images",
            "trees": self.L / "trees",
        }

    def cam_label(self, c) -> str:
        if c == FUSION:
            return "Füzyon (4K)"
        return self.camera_labels.get(int(c), f"K{c}")

    def run_list(self):
        return [(m, c, k, f"{Path(m).stem}_cam{c}_fold{k}")
                for m in self.models for c in self.cameras for k in range(self.n_folds)]


LOG = logging.getLogger("tomato")


# ----------------------------------------------------------------------------
# Genel yardımcılar
# ----------------------------------------------------------------------------
def setup(cfg: Config):
    for d in cfg.dirs().values():
        d.mkdir(parents=True, exist_ok=True)
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(cfg.dirs()["logs"] / "pipeline.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    (cfg.dirs()["logs"] / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))
    env = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for mod in ("ultralytics", "torch", "torchvision", "sklearn", "huggingface_hub", "onnxruntime", "numpy", "pandas"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["gpu"] = None
    try:
        env["cpu"] = subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    (cfg.dirs()["logs"] / "environment.json").write_text(json.dumps(env, indent=2))
    LOG.info("Ortam: %s", env)
    return env


def _atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"date": str})


def _hf_token():
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


# ----------------------------------------------------------------------------
# AŞAMA 1: indeks + veri denetimi
# ----------------------------------------------------------------------------
def _norm_relpath(fp: str) -> str:
    fp = fp.replace("\\", "/")
    if "/data/" in fp:
        fp = "data/" + fp.split("/data/", 1)[1]
    while fp.startswith("./"):
        fp = fp[2:]
    return fp.lstrip("/")


def _parse_date(sample: dict, m):
    if m:
        return m.group(5)[:8]
    v = sample.get("capture_datetime")
    if isinstance(v, dict):
        v = v.get("$date")
    if isinstance(v, (int, float)):
        return time.strftime("%Y%m%d", time.gmtime(v / 1000))
    if isinstance(v, str):
        return re.sub(r"[^0-9]", "", v)[:8]
    return None


def stage_index(cfg: Config) -> pd.DataFrame:
    d = cfg.dirs()
    out = d["cache"] / "metadata_full.csv"
    if out.exists():
        df = _read(out)
        LOG.info("metadata_full.csv mevcut (%d görüntü).", len(df))
        return df
    sj = d["cache"] / "samples.json"
    if not sj.exists():
        if cfg.offline_src:
            shutil.copy(Path(cfg.offline_src) / "samples.json", sj)
        else:
            from huggingface_hub import hf_hub_download
            LOG.info("samples.json indiriliyor (~218 MB)...")
            p = hf_hub_download(cfg.hf_repo, "samples.json", repo_type="dataset",
                                local_dir=str(d["cache"]), token=_hf_token())
            if Path(p) != sj:
                shutil.move(p, sj)
    with open(sj, "r", encoding="utf-8") as f:
        obj = json.load(f)
    samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
    if not isinstance(samples, list):
        raise RuntimeError("samples.json yapısı beklenmedik.")

    rows, skipped = [], 0
    for s in samples:
        if "det" not in (s.get("tags") or []):
            continue
        fp = _norm_relpath(s.get("filepath", ""))
        m = FNAME_RE.search(Path(fp).name)
        cam = int(m.group(1)) if m else s.get("piid")
        plant = int(m.group(3)) if m else s.get("plant_id")
        pose = int(m.group(4)) if m else s.get("pose_id")
        bbch = s.get("bbch_stage")
        if bbch is None:
            mm = re.search(r"(\d+)", (s.get("classification") or {}).get("label", "") or "")
            bbch = int(mm.group(1)) if mm else None
        date = _parse_date(s, m)
        if None in (cam, plant, pose, bbch, date):
            skipped += 1
            continue
        rows.append(dict(relpath=fp, camera=int(cam), plant=int(plant), pose=int(pose), date=str(date), bbch=int(bbch)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Hiç 'det' örneği ayrıştırılamadı.")
    df["stage"] = df["bbch"] // 10
    bad = ~df["stage"].isin(STAGE_TO_CLASS)
    if bad.any():
        LOG.warning("Beklenmeyen ana evreli %d görüntü çıkarıldı: %s", bad.sum(), sorted(df.loc[bad, "bbch"].unique()))
        df = df[~bad].copy()
    df["cls"] = df["stage"].map(STAGE_TO_CLASS)
    df["session"] = "p" + df["plant"].map("{:03d}".format) + "_" + df["date"]
    df["image_id"] = ("c" + df["camera"].astype(str) + "_" + df["session"] + "_o" + df["pose"].map("{:02d}".format))
    dup = df["image_id"].duplicated(keep=False)
    if dup.any():
        LOG.warning("%d görüntü aynı (kamera, bitki, tarih, poz) anahtarını paylaşıyor; ilk kayıt tutuldu.", dup.sum())
        df = df.drop_duplicates("image_id", keep="first")

    if not cfg.offline_src:
        try:
            from huggingface_hub import HfApi
            files = set(HfApi(token=_hf_token()).list_repo_files(cfg.hf_repo, repo_type="dataset"))
            miss = ~df["relpath"].isin(files)
            if miss.any():
                by_base = {Path(f).name: f for f in files}
                df.loc[miss, "relpath"] = df.loc[miss, "relpath"].map(lambda p: by_base.get(Path(p).name, p))
                still = ~df["relpath"].isin(files)
                LOG.warning("Yol eşleşmesi: %d düzeltildi, %d eksik (çıkarıldı).", int(miss.sum() - still.sum()), int(still.sum()))
                df = df[~still]
        except Exception as e:
            LOG.warning("Depo dosya listesi alınamadı (%s); doğrulama atlandı.", e)

    df = df.reset_index(drop=True)
    _atomic_csv(df, out)
    audit = {
        "n_images": int(len(df)), "n_skipped_unparsed": int(skipped), "expected_images": 64464,
        "n_plants": int(df.plant.nunique()), "n_sessions": int(df.session.nunique()),
        "cameras": sorted(map(int, df.camera.unique())), "poses": sorted(map(int, df.pose.unique())),
        "sessions_with_multiple_bbch": int((df.groupby("session").bbch.nunique() > 1).sum()),
        "images_per_session": df.groupby("session").size().describe().to_dict(),
        "bbch_codes": sorted(map(int, df.bbch.unique())),
    }
    (d["tables"] / "T0_metadata_audit.json").write_text(json.dumps(audit, indent=2, default=_json_default))
    LOG.info("Veri denetimi: %s", audit)
    return df


# ----------------------------------------------------------------------------
# AŞAMA 2: oturum başına poz seçimi
# ----------------------------------------------------------------------------
def _pick(seed: int, key: str, options: list[int]) -> int:
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16)
    return sorted(options)[h % len(options)]


def stage_select(cfg: Config) -> pd.DataFrame:
    d = cfg.dirs()
    man_p, sub_p = d["manifests"] / "selection_manifest.csv", d["manifests"] / "subset_images.csv"
    if man_p.exists() and sub_p.exists():
        LOG.info("Seçim manifesti mevcut.")
        return _read(sub_p)
    df = _read(d["cache"] / "metadata_full.csv")
    df = df[df.camera.isin(cfg.cameras)]
    rows, excluded = [], []
    for sess, g in df.groupby("session", sort=True):
        per_cam = [set(g.loc[g.camera == c, "pose"]) for c in cfg.cameras]
        common = sorted(set.intersection(*per_cam)) if all(per_cam) else []
        if not common:
            excluded.append((sess, "dört kamerada ortak poz yok"))
            continue
        bb = g["bbch"].mode().iloc[0]
        if g["bbch"].nunique() > 1:
            LOG.warning("%s oturumunda birden fazla BBCH; mod değeri (%d) kullanıldı.", sess, bb)
        pose = _pick(cfg.pose_seed, sess, common)
        row = dict(session=sess, plant=int(g.plant.iloc[0]), date=g.date.iloc[0], bbch=int(bb),
                   cls=STAGE_TO_CLASS[bb // 10], n_common_poses=len(common), pose=pose)
        for c in cfg.cameras:
            r = g[(g.camera == c) & (g.pose == pose)].iloc[0]
            row[f"image_id_c{c}"], row[f"relpath_c{c}"] = r.image_id, r.relpath
        rows.append(row)
    man = pd.DataFrame(rows)
    _atomic_csv(man, man_p)
    sub = []
    for r in man.itertuples():
        for c in cfg.cameras:
            sub.append(dict(image_id=getattr(r, f"image_id_c{c}"), relpath=getattr(r, f"relpath_c{c}"), camera=c,
                            session=r.session, plant=r.plant, date=r.date, pose=r.pose, bbch=r.bbch, cls=r.cls))
    sub = pd.DataFrame(sub)
    _atomic_csv(sub, sub_p)
    if excluded:
        _atomic_csv(pd.DataFrame(excluded, columns=["session", "reason"]), d["manifests"] / "excluded_sessions.csv")
    LOG.info("Seçilen oturum: %d | alt küme görüntü: %d | dışlanan oturum: %d (poz tohumu %d)",
             len(man), len(sub), len(excluded), cfg.pose_seed)
    return sub


# ----------------------------------------------------------------------------
# AŞAMA 3: görüntüler
# ----------------------------------------------------------------------------
def _prep_one(src: Path, dst: Path, size: int):
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (0, 0, 0))
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        canvas.resize((size, size), Image.BICUBIC).save(dst, "JPEG", quality=95)


def _download(cfg: Config, rels: list[str], dest: Path):
    if cfg.offline_src:
        for r in rels:
            (dest / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(cfg.offline_src) / r, dest / r)
        return
    from huggingface_hub import snapshot_download
    last_err = None
    for attempt in range(5):
        try:
            snapshot_download(cfg.hf_repo, repo_type="dataset", allow_patterns=rels, local_dir=str(dest),
                              max_workers=cfg.download_workers, token=_hf_token())
            if all((dest / r).exists() for r in rels):
                return
            last_err = RuntimeError("seçmeli indirme sonrası dosyalar eksik")
        except Exception as e:
            last_err = e
        wait = 30 * (2 ** attempt)
        LOG.warning("Seçmeli indirme başarısız (%s); %d sn sonra yeniden denenecek.", last_err, wait)
        time.sleep(wait)
    if not cfg.allow_full_download:
        raise RuntimeError(f"Seçmeli indirme başarısız: {last_err}")
    LOG.warning("Seçmeli indirme başarısız; TAM indirmeye geçiliyor (~49 GB). Yeterli disk alanı olduğundan emin olun.")
    full = cfg.dirs()["raw"] / "_full"
    snapshot_download(cfg.hf_repo, repo_type="dataset", local_dir=str(full),
                      max_workers=cfg.download_workers, token=_hf_token())
    for r in rels:
        (dest / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(full / r, dest / r)


def stage_images(cfg: Config):
    d = cfg.dirs()
    sub = _read(d["manifests"] / "subset_images.csv").sort_values("image_id").reset_index(drop=True)
    n_sh = int(np.ceil(len(sub) / cfg.shard_size))
    for k in range(n_sh):
        tar_p, done = d["shards"] / f"shard_{k:03d}.tar", d["shards"] / f"shard_{k:03d}.done"
        if tar_p.exists() and done.exists():
            continue
        part = sub.iloc[k * cfg.shard_size:(k + 1) * cfg.shard_size]
        LOG.info("Parça %d/%d: %d görüntü indiriliyor...", k + 1, n_sh, len(part))
        raw, prep = d["raw"] / f"shard_{k:03d}", d["raw"] / f"prep_{k:03d}"
        raw.mkdir(parents=True, exist_ok=True)
        prep.mkdir(parents=True, exist_ok=True)
        _download(cfg, part["relpath"].tolist(), raw)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda r: _prep_one(raw / r.relpath, prep / f"{r.image_id}.jpg", cfg.prep_size), part.itertuples()))
        n_ok = len(list(prep.glob("*.jpg")))
        if n_ok != len(part):
            raise RuntimeError(f"Parça {k}: {len(part)} yerine {n_ok} görüntü hazırlandı.")
        local_tar = d["raw"] / f"shard_{k:03d}.tar"
        with tarfile.open(local_tar, "w") as tf:
            for p in sorted(prep.glob("*.jpg")):
                tf.add(p, arcname=p.name)
        shutil.copy(local_tar, tar_p)
        done.write_text(str(len(part)))
        for p in (raw, prep):
            shutil.rmtree(p, ignore_errors=True)
        local_tar.unlink(missing_ok=True)
    shutil.rmtree(d["raw"] / "_full", ignore_errors=True)
    restore_images(cfg)


def restore_images(cfg: Config):
    d = cfg.dirs()
    sub = _read(d["manifests"] / "subset_images.csv")
    img = d["images"]
    img.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in img.glob("*.jpg")}
    if set(sub.image_id) <= have:
        return
    for t in sorted(d["shards"].glob("shard_*.tar")):
        with tarfile.open(t) as tf:
            tf.extractall(img)
    missing = set(sub.image_id) - {p.stem for p in img.glob("*.jpg")}
    if missing:
        raise RuntimeError(f"{len(missing)} görüntü eksik; stage_images'ı yeniden çalıştırın.")
    LOG.info("Yerel diske açılan görüntü: %d", len(sub))


# ----------------------------------------------------------------------------
# AŞAMA 4: bitki düzeyinde katlar
# ----------------------------------------------------------------------------
def stage_folds(cfg: Config) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold
    d = cfg.dirs()
    fp = d["manifests"] / "plant_folds.csv"
    man = _read(d["manifests"] / "selection_manifest.csv")
    if not fp.exists():
        y, g = man["cls"].values, man["plant"].values
        chosen = None
        for t in range(1000):
            rs = cfg.fold_seed + t
            sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=rs)
            fold = np.full(len(man), -1)
            for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
                fold[te] = k
            if all(set(y[fold == k]) == set(CLASSES) for k in range(cfg.n_folds)):
                chosen = (rs, t, fold)
                break
        if chosen is None:
            raise RuntimeError("Her katta 6 sınıfı içeren bölme bulunamadı.")
        rs, t, fold = chosen
        pf = (pd.DataFrame({"plant": g, "fold": fold}).drop_duplicates().sort_values("plant"))
        assert pf.plant.is_unique, "Bir bitki birden fazla kata atanmış."
        pf["random_state"] = rs
        _atomic_csv(pf, fp)
        LOG.info("Kat manifesti oluşturuldu: random_state=%d (%d. deneme; tohum %d).", rs, t + 1, cfg.fold_seed)
    pf = _read(fp)
    m = man.merge(pf[["plant", "fold"]], on="plant", validate="many_to_one")
    rows = []
    for k in range(cfg.n_folds):
        q = m[m.fold == k]
        row = dict(fold=k, n_plants=q.plant.nunique(), n_sessions=len(q))
        for c in CLASSES:
            row[f"sessions_{CLASS_SHORT[c]}"] = int((q.cls == c).sum())
        rows.append(row)
    t2 = pd.DataFrame(rows)
    missing = [(r.fold, c) for r in t2.itertuples() for c in CLASSES if getattr(r, f"sessions_{CLASS_SHORT[c]}") == 0]
    if missing:
        raise RuntimeError(f"Sınıfı eksik kat(lar): {missing}")
    _atomic_csv(t2, d["tables"] / "T2_folds.csv")
    LOG.info("\n%s", t2.to_string(index=False))
    return t2


def _split_of(fold: int, k: int, n: int) -> str:
    if fold == k:
        return "test"
    if fold == (k + 1) % n:
        return "val"
    return "train"


def build_tree(cfg: Config, camera: int, k: int) -> Path:
    root = cfg.dirs()["trees"] / f"cam{camera}_fold{k}"
    if (root / ".ok").exists():
        return root
    shutil.rmtree(root, ignore_errors=True)
    d = cfg.dirs()
    sub = _read(d["manifests"] / "subset_images.csv").merge(_read(d["manifests"] / "plant_folds.csv")[["plant", "fold"]], on="plant")
    sub = sub[sub.camera == camera]
    for part in ("train", "val", "test"):
        for c in CLASSES:
            (root / part / c).mkdir(parents=True, exist_ok=True)
    for r in sub.itertuples():
        os.symlink(d["images"] / f"{r.image_id}.jpg", root / _split_of(r.fold, k, cfg.n_folds) / r.cls / f"{r.image_id}.jpg")
    (root / ".ok").write_text("ok")
    return root


# ----------------------------------------------------------------------------
# AŞAMA 5: eğitim + test-katı tahmini
# ----------------------------------------------------------------------------
def _predict(cfg: Config, weights: Path, image_ids: list[str], device=None) -> np.ndarray:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    paths = [str(cfg.dirs()["images"] / f"{i}.jpg") for i in image_ids]
    probs, names = [], None
    for i in range(0, len(paths), 128):
        res = model.predict(paths[i:i + 128], imgsz=cfg.imgsz, batch=128, verbose=False, device=device)
        probs += [r.probs.data.cpu().numpy() for r in res]
        names = res[0].names
    P = np.vstack(probs)
    order = [CLASSES.index(names[j]) for j in range(len(names))]
    out = np.zeros_like(P)
    out[:, order] = P
    return out


def stage_train_eval(cfg: Config, only: list[str] | None = None):
    import torch
    from ultralytics import YOLO
    restore_images(cfg)
    d = cfg.dirs()
    sub = _read(d["manifests"] / "subset_images.csv").merge(_read(d["manifests"] / "plant_folds.csv")[["plant", "fold"]], on="plant")
    for model_name, cam, k, run in cfg.run_list():
        if only and run not in only:
            continue
        rdir = d["runs"] / run
        if (rdir / "DONE").exists():
            continue
        tree = build_tree(cfg, cam, k)
        last, best = rdir / "weights" / "last.pt", rdir / "weights" / "best.pt"
        t0 = time.time()
        if not (rdir / "TRAIN_DONE").exists():
            if last.exists():
                LOG.info("[%s] yarıda kalan eğitim sürdürülüyor.", run)
                try:
                    YOLO(str(last)).train(resume=True)
                except Exception as e:
                    msg = str(e).lower()
                    if not ("nothing to resume" in msg or "is finished" in msg):
                        raise
            else:
                LOG.info("[%s] eğitim başlıyor.", run)
                YOLO(model_name).train(
                    data=str(tree), imgsz=cfg.imgsz, epochs=cfg.epochs, patience=cfg.patience, batch=cfg.batch,
                    workers=cfg.workers, seed=cfg.train_seed, deterministic=True, project=str(d["runs"]),
                    name=run, exist_ok=True, pretrained=True, plots=True, cache="ram", verbose=False)
            (rdir / "TRAIN_DONE").write_text(f"{time.time() - t0:.0f}")
        if not best.exists():
            raise RuntimeError(f"[{run}] best.pt yok.")
        te = sub[(sub.camera == cam) & (sub.fold == k)].reset_index(drop=True)
        P = _predict(cfg, best, te.image_id.tolist())
        pred = te[["image_id", "session", "plant", "fold", "camera", "bbch", "cls"]].copy()
        for j, c in enumerate(CLASSES):
            pred[f"p_{c}"] = P[:, j]
        pred["pred"] = [CLASSES[j] for j in P.argmax(1)]
        _atomic_csv(pred, rdir / "test_predictions.csv")
        (rdir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        LOG.info("[%s] bitti | test doğruluğu %.3f | süre %.0f sn", run, (pred.pred == pred.cls).mean(), time.time() - t0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def status(cfg: Config) -> pd.DataFrame:
    rows = []
    for m, c, k, run in cfg.run_list():
        r = cfg.dirs()["runs"] / run
        ep = len(pd.read_csv(r / "results.csv")) if (r / "results.csv").exists() else 0
        rows.append(dict(run=run, epochs_logged=ep, train_done=(r / "TRAIN_DONE").exists(), done=(r / "DONE").exists()))
    s = pd.DataFrame(rows)
    LOG.info("Tamamlanan: %d / %d", s.done.sum(), len(s))
    return s


# ----------------------------------------------------------------------------
# AŞAMA 6: OOF birleştirme + geç füzyon
# ----------------------------------------------------------------------------
PCOLS = [f"p_{c}" for c in CLASSES]


def stage_oof(cfg: Config):
    d = cfg.dirs()
    n_sessions = len(_read(d["manifests"] / "selection_manifest.csv"))
    for m in cfg.models:
        stem = Path(m).stem
        per_cam = {}
        for c in cfg.cameras:
            parts = []
            for k in range(cfg.n_folds):
                f = d["runs"] / f"{stem}_cam{c}_fold{k}" / "test_predictions.csv"
                if not f.exists():
                    raise RuntimeError(f"Eksik koşu: {f.parent.name}")
                parts.append(_read(f))
            oof = pd.concat(parts, ignore_index=True)
            if oof.session.duplicated().any() or len(oof) != n_sessions:
                raise RuntimeError(f"{stem} K{c}: OOF {len(oof)} satır, beklenen {n_sessions} tekil oturum.")
            _atomic_csv(oof, d["oof"] / f"oof_{stem}_cam{c}.csv")
            per_cam[c] = oof.set_index("session")
        base = per_cam[cfg.cameras[0]][["plant", "fold", "bbch", "cls"]].copy()
        P = np.mean([per_cam[c].loc[base.index, PCOLS].values for c in cfg.cameras], axis=0)
        fus = base.copy()
        for j, col in enumerate(PCOLS):
            fus[col] = P[:, j]
        fus["pred"] = [CLASSES[j] for j in P.argmax(1)]
        fus["camera"] = FUSION
        _atomic_csv(fus.reset_index(), d["oof"] / f"oof_{stem}_{FUSION}.csv")
        LOG.info("%s: OOF ve füzyon hazır (%d oturum).", stem, len(fus))


def _load_oof(cfg: Config) -> dict:
    out = {}
    for m in cfg.models:
        stem = Path(m).stem
        for v in list(cfg.cameras) + [FUSION]:
            f = cfg.dirs()["oof"] / f"oof_{stem}_{'cam' + str(v) if v != FUSION else FUSION}.csv"
            if f.exists():
                out[(stem, v)] = _read(f)
    return out


# ----------------------------------------------------------------------------
# AŞAMA 7: verimlilik (tek kamera vs dört kamera)
# ----------------------------------------------------------------------------
def stage_efficiency(cfg: Config) -> pd.DataFrame:
    import torch
    from ultralytics import YOLO
    d = cfg.dirs()
    out_p = d["tables"] / "T7_cost.csv"
    if out_p.exists():
        return _read(out_p)
    restore_images(cfg)
    man = _read(d["manifests"] / "selection_manifest.csv")
    # Tek bir görüntüyü art arda zamanlamak önbellek etkisini büyütebilir. Bu nedenle
    # farklı bitki ve evreleri kapsayan deterministik bir oturum örneklemi kullanılır.
    n_latency_sessions = min(cfg.latency_runs, len(man))
    per_class_n = max(1, int(np.ceil(n_latency_sessions / K)))
    latency_sessions = pd.concat([
        x.sample(n=min(len(x), per_class_n), random_state=cfg.bootstrap_seed)
        for _, x in man.groupby("cls", sort=True)
    ], ignore_index=True).drop_duplicates("session").head(n_latency_sessions)
    if len(latency_sessions) < n_latency_sessions:
        extra = man[~man.session.isin(latency_sessions.session)].sample(
            n=n_latency_sessions - len(latency_sessions), random_state=cfg.bootstrap_seed)
        latency_sessions = pd.concat([latency_sessions, extra], ignore_index=True)
    rows = []
    for m in cfg.models:
        stem = Path(m).stem
        W = [d["runs"] / f"{stem}_cam{c}_fold0" / "weights" / "best.pt" for c in cfg.cameras]
        if not all(w.exists() for w in W):
            LOG.warning("%s ağırlıkları eksik; verimlilik atlandı.", stem)
            continue
        base = YOLO(str(W[0]))
        params_single = sum(p.numel() for p in base.model.parameters()) / 1e6
        row = dict(model=stem,
                   latency_n_sessions=int(len(latency_sessions)),
                   weights_MB_single=W[0].stat().st_size / 1e6,
                   weights_MB_fusion=sum(w.stat().st_size for w in W) / 1e6,
                   params_M_single=params_single,
                   params_M_fusion=4 * params_single)
        try:
            from ultralytics.utils.torch_utils import get_flops
            row["GFLOPs_per_image"] = float(get_flops(base.model, cfg.imgsz))
            row["GFLOPs_fusion_per_session"] = 4 * row["GFLOPs_per_image"]
        except Exception as e:
            LOG.warning("GFLOPs hesaplanamadı: %s", e)

        # Görüntüler zamanlamadan ÖNCE belleğe yüklenir (BGR numpy, Ultralytics/cv2 kuralı).
        # Böylece ölçüm disk okuma ve JPEG çözme süresini değil, model çıkarımını yansıtır;
        # tek kamera ile füzyon arasında soğuk/sıcak disk önbelleği farkı oluşmaz.
        import cv2
        mem = {(r.session, c): cv2.imread(str(d["images"] / f"{getattr(r, f'image_id_c{c}')}.jpg"))
               for r in latency_sessions.itertuples() for c in cfg.cameras}
        if any(v is None for v in mem.values()):
            raise RuntimeError("Gecikme örneklemindeki bazı görüntüler okunamadı.")

        def timeit(models, camera_ids, device):
            warm = latency_sessions.head(min(10, len(latency_sessions)))
            for sess in warm.session:
                for mdl, cam in zip(models, camera_ids):
                    mdl.predict(mem[(sess, cam)], imgsz=cfg.imgsz, device=device, verbose=False)
            ts = []
            for sess in latency_sessions.session:
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = time.perf_counter()
                for mdl, cam in zip(models, camera_ids):
                    mdl.predict(mem[(sess, cam)], imgsz=cfg.imgsz, device=device, verbose=False)
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000)
            return float(np.median(ts)), float(np.percentile(ts, 90))

        devices = ([("gpu", 0)] if torch.cuda.is_available() else []) + [("cpu", "cpu")]
        for tag, dev in devices:
            ms = [YOLO(str(w)) for w in W]
            s_med, s_p90 = timeit(ms[:1], cfg.cameras[:1], dev)
            f_med, f_p90 = timeit(ms, cfg.cameras, dev)
            row.update({f"{tag}_single_ms_median": s_med, f"{tag}_single_ms_p90": s_p90,
                        f"{tag}_fusion_ms_median": f_med, f"{tag}_fusion_ms_p90": f_p90,
                        f"{tag}_single_sessions_per_s": 1000 / s_med, f"{tag}_fusion_sessions_per_s": 1000 / f_med})
        try:
            onnx = [YOLO(str(w)).export(format="onnx", imgsz=cfg.imgsz, dynamic=False) for w in W]
            ms = [YOLO(p, task="classify") for p in onnx]
            s_med, s_p90 = timeit(ms[:1], cfg.cameras[:1], "cpu")
            f_med, f_p90 = timeit(ms, cfg.cameras, "cpu")
            row.update({"onnxcpu_single_ms_median": s_med, "onnxcpu_single_ms_p90": s_p90,
                        "onnxcpu_fusion_ms_median": f_med, "onnxcpu_fusion_ms_p90": f_p90,
                        "onnxcpu_single_sessions_per_s": 1000 / s_med, "onnxcpu_fusion_sessions_per_s": 1000 / f_med})
        except Exception as e:
            LOG.warning("ONNX ölçümü başarısız: %s", e)
        rows.append(row)
        LOG.info("Maliyet: %s", row)
    df = pd.DataFrame(rows)
    _atomic_csv(df, out_p)
    return df


# ----------------------------------------------------------------------------
# AŞAMA 8: istatistik ve analiz
# ----------------------------------------------------------------------------
def _cm(y, p) -> np.ndarray:
    yi = np.array([CLASSES.index(c) for c in y])
    pi = np.array([CLASSES.index(c) for c in p])
    M = np.zeros((K, K), dtype=np.int64)
    np.add.at(M, (yi, pi), 1)
    return M


def _metrics_from_cm(M: np.ndarray) -> dict:
    """M: (..., K, K) gerçek x tahmin. Destek ve tahmini olmayan sınıflar ortalamadan çıkarılır."""
    M = M.astype(float)
    tp = np.diagonal(M, axis1=-2, axis2=-1)
    sup, prd = M.sum(-1), M.sum(-2)
    denom = sup + prd
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
        rec = np.where(sup > 0, tp / sup, np.nan)
        n = M.sum((-2, -1))
        acc = tp.sum(-1) / n
    return {"macro_f1": np.nanmean(f1, -1), "balanced_accuracy": np.nanmean(rec, -1), "accuracy": acc}


def _ordinal(y, p) -> dict:
    yi = np.array([CLASSES.index(c) for c in y])
    pi = np.array([CLASSES.index(c) for c in p])
    e = np.abs(yi - pi)
    return {"ordinal_mae": float(e.mean()), "adjacent_error_share": float((e[e > 0] == 1).mean()) if (e > 0).any() else np.nan}


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def _boot_p(diff: np.ndarray) -> float:
    """İki yönlü bootstrap p değeri. Hiç ters yönlü örnek yoksa değer 1/B'ye sabitlenir;
    bu bir ölçüm değil çözünürlük sınırıdır. Makalede "p = 1/B" değil "p < 0,001" (B=2000)
    yazılır. Holm düzeltmesi bu tabanı çarptığı için, taban kaynaklı düzeltilmiş değerler de
    "≤" ile raporlanır (bkz. p_boot_at_floor sütunu)."""
    diff = diff[~np.isnan(diff)]
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(diff))))


def _savefig(fig, name, cfg):
    for ext in ("png", "pdf"):
        fig.savefig(cfg.dirs()["figures"] / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def stage_analysis(cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    d = cfg.dirs()
    T = d["tables"]
    tables: dict[str, pd.DataFrame] = {}
    man = _read(d["manifests"] / "selection_manifest.csv")
    full = _read(d["cache"] / "metadata_full.csv")

    # ---- T1 veri seti ve sınıf eşlemesi
    t1 = []
    for c in CLASSES:
        q, qf = man[man.cls == c], full[full.cls == c]
        t1.append(dict(sinif=CLASS_SHORT[c], evre=CLASS_LABELS_TR[c],
                       bbch_kodlari=", ".join(map(str, sorted(qf.bbch.unique()))),
                       bitki=q.plant.nunique(), oturum=len(q), altkume_goruntu=len(q) * len(cfg.cameras),
                       tam_veri_goruntu=len(qf)))
    t1 = pd.DataFrame(t1)
    t1.loc[len(t1)] = dict(sinif="Toplam", evre="", bbch_kodlari="", bitki=man.plant.nunique(), oturum=len(man),
                           altkume_goruntu=len(man) * len(cfg.cameras), tam_veri_goruntu=len(full))
    tables["T1_dataset"] = t1
    if (T / "T2_folds.csv").exists():
        tables["T2_folds"] = _read(T / "T2_folds.csv")

    # ---- F1 sınıf dağılımı + evre örnekleri, F2 bir oturumun dört görünümü
    from PIL import Image
    fig, ax = plt.subplots(figsize=(6, 2.6))
    tt = t1.iloc[:-1]
    ax.bar([CLASS_LABELS_TR[c] for c in CLASSES], tt.oturum, color="#4C72B0")
    for i, v in enumerate(tt.oturum):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Oturum sayısı")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _savefig(fig, "F1a_class_distribution", cfg)
    plt.close(fig)
    try:
        cam0 = cfg.cameras[0]
        fig, axes = plt.subplots(1, K, figsize=(10, 2.1))
        for ax, c in zip(axes, CLASSES):
            r = man[man.cls == c].sample(1, random_state=1).iloc[0]
            ax.imshow(Image.open(d["images"] / f"{r[f'image_id_c{cam0}']}.jpg"))
            ax.set_title(CLASS_LABELS_TR[c], fontsize=7)
            ax.axis("off")
        _savefig(fig, "F1b_stage_examples", cfg)
        plt.close(fig)
        r = man[man.cls == "3_S6"].sample(1, random_state=2).iloc[0]
        fig, axes = plt.subplots(1, len(cfg.cameras), figsize=(2.2 * len(cfg.cameras), 2.4))
        for ax, c in zip(np.atleast_1d(axes), cfg.cameras):
            ax.imshow(Image.open(d["images"] / f"{r[f'image_id_c{c}']}.jpg"))
            ax.set_title(cfg.cam_label(c), fontsize=8)
            ax.axis("off")
        _savefig(fig, "F2_session_four_views", cfg)
        plt.close(fig)
    except Exception as e:
        LOG.warning("Örnek görüntü şekilleri üretilemedi: %s", e)

    oof = _load_oof(cfg)
    if not oof:
        _write_xlsx(tables, cfg)
        LOG.warning("OOF tahmini yok; yalnızca veri tabloları üretildi.")
        return
    models = sorted({m for m, _ in oof})
    views = [v for v in list(cfg.cameras) + [FUSION] if any((m, v) in oof for m in models)]

    # ---- bitki kümeli bootstrap altyapısı (tüm sistemler aynı örneklerle -> eşleştirilmiş)
    plants = np.array(sorted(man.plant.unique()))
    pidx = {p: i for i, p in enumerate(plants)}
    rng = np.random.default_rng(cfg.bootstrap_seed)
    Wb = rng.multinomial(len(plants), np.full(len(plants), 1 / len(plants)), size=cfg.bootstrap_B)  # B x P
    plant_cms, boot = {}, {}
    for key, df in oof.items():
        C = np.zeros((len(plants), K, K), dtype=np.int64)
        for p, g in df.groupby("plant"):
            C[pidx[p]] = _cm(g.cls, g.pred)
        plant_cms[key] = C
        boot[key] = _metrics_from_cm(np.tensordot(Wb, C, axes=(1, 0)))

    # ---- T3 ana sonuçlar
    t3 = []
    for m in models:
        for v in views:
            if (m, v) not in oof:
                continue
            df = oof[(m, v)]
            pt = _metrics_from_cm(plant_cms[(m, v)].sum(0))
            row = dict(model=m, view=cfg.cam_label(v), view_key=str(v), n_sessions=len(df))
            for met in ("macro_f1", "balanced_accuracy", "accuracy"):
                lo, hi = np.nanpercentile(boot[(m, v)][met], [2.5, 97.5])
                row.update({met: float(pt[met]), f"{met}_ci_lo": lo, f"{met}_ci_hi": hi})
            row["kappa"] = cohen_kappa_score(df.cls, df.pred)
            row.update(_ordinal(df.cls, df.pred))
            t3.append(row)
    t3 = pd.DataFrame(t3)
    tables["T3_main_results"] = t3
    fmt = t3[["model", "view"]].copy()
    for met in ("macro_f1", "balanced_accuracy", "accuracy"):
        fmt[met] = t3.apply(lambda r: f"{r[met]:.3f} [{r[met + '_ci_lo']:.3f}–{r[met + '_ci_hi']:.3f}]", axis=1)
    for met in ("kappa", "ordinal_mae", "adjacent_error_share"):
        fmt[met] = t3[met].map("{:.3f}".format)
    tables["T3b_main_results_formatted"] = fmt

    # ---- T3c kat bazında makro F1
    rows = []
    for (m, v), df in oof.items():
        for k, g in df.groupby("fold"):
            rows.append(dict(model=m, view=cfg.cam_label(v), fold=int(k), **{x: float(y) for x, y in _metrics_from_cm(_cm(g.cls, g.pred)).items()}))
    pf = pd.DataFrame(rows)
    tables["T3c_per_fold"] = pf
    tables["T3d_per_fold_summary"] = (pf.groupby(["model", "view"])[["macro_f1", "balanced_accuracy"]]
                                      .agg(["mean", "std"]).pipe(lambda x: x.set_axis([f"{a}_{b}" for a, b in x.columns], axis=1)).reset_index())

    # ---- T4 sınıf bazlı P/R/F1
    rows = []
    for (m, v), df in oof.items():
        P, R, F, S = precision_recall_fscore_support(df.cls, df.pred, labels=CLASSES, zero_division=0)
        for j, c in enumerate(CLASSES):
            rows.append(dict(model=m, view=cfg.cam_label(v), sinif=CLASS_SHORT[c], precision=P[j], recall=R[j], f1=F[j], support=int(S[j])))
    t4 = pd.DataFrame(rows)
    tables["T4_per_class"] = t4

    # ---- T5/T6 eşleştirilmiş karşılaştırmalar (Holm)
    def compare(pairs, family):
        out = []
        for m_a, v_a, m_b, v_b in pairs:
            if (m_a, v_a) not in oof or (m_b, v_b) not in oof:
                continue
            row = dict(family=family, A=f"{m_a} {cfg.cam_label(v_a)}", B=f"{m_b} {cfg.cam_label(v_b)}")
            for met in ("macro_f1", "balanced_accuracy"):
                pa = _metrics_from_cm(plant_cms[(m_a, v_a)].sum(0))[met]
                pb = _metrics_from_cm(plant_cms[(m_b, v_b)].sum(0))[met]
                diff = boot[(m_b, v_b)][met] - boot[(m_a, v_a)][met]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                row.update({f"{met}_A": pa, f"{met}_B": pb, f"d_{met}(B-A)": pb - pa,
                            f"d_{met}_ci_lo": lo, f"d_{met}_ci_hi": hi})
                if met == "macro_f1":
                    row["p_boot"] = _boot_p(diff)
            out.append(row)
        df = pd.DataFrame(out)
        if not df.empty:
            df["p_boot_at_floor"] = np.isclose(df["p_boot"], 1.0 / cfg.bootstrap_B)
            df["p_holm"] = _holm(df["p_boot"].values)
            df["significant_0.05"] = df["p_holm"] < 0.05
        return df

    cams = list(cfg.cameras)
    t5 = pd.concat([compare([(m, a, m, b) for i, a in enumerate(cams) for b in cams[i + 1:]], f"{m}: kamera çiftleri")
                    for m in models], ignore_index=True)
    tables["T5_camera_pairs"] = t5
    t6a = pd.concat([compare([(m, c, m, FUSION) for c in cams], f"{m}: füzyon - tek kamera") for m in models], ignore_index=True)
    tables["T6a_fusion_vs_single"] = t6a
    if len(models) == 2:
        t6b = compare([(models[0], v, models[1], v) for v in views], f"{models[1]} - {models[0]}")
        tables["T6b_model_comparison"] = t6b

    # ---- T7 maliyet–başarı
    if (T / "T7_cost.csv").exists():
        cost = _read(T / "T7_cost.csv")
        tables["T7_cost"] = cost
        rows = []
        for m in models:
            sub = t3[(t3.model == m) & (t3.view_key != FUSION)]
            if sub.empty or (m, FUSION) not in oof:
                continue
            best = sub.loc[sub.macro_f1.idxmax()]
            fus = t3[(t3.model == m) & (t3.view_key == FUSION)].iloc[0]
            c = cost[cost.model == m]
            # NOT: Series.view bir pandas metodudur; satır değerleri köşeli parantezle okunmalı.
            row = dict(model=m, best_single=best["view"], macro_f1_best_single=float(best["macro_f1"]),
                       macro_f1_fusion=float(fus["macro_f1"]), d_macro_f1=float(fus["macro_f1"] - best["macro_f1"]),
                       images_per_session_single=1, images_per_session_fusion=len(cfg.cameras))
            if not c.empty:
                for col in c.columns:
                    if col.endswith("_ms_median") or col.startswith("GFLOPs"):
                        row[col] = float(c.iloc[0][col])
            rows.append(row)
        tables["T7b_cost_accuracy_tradeoff"] = pd.DataFrame(rows)

    # ================= ŞEKİLLER =================
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    # F3 görünüm bazında makro F1 (bitki kümeli %95 GA)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        q = t3[t3.model == m].set_index("view_key").reindex([str(v) for v in views])
        x = np.arange(len(views)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, q.macro_f1, w, color=colors[i], label=m,
               yerr=[q.macro_f1 - q.macro_f1_ci_lo, q.macro_f1_ci_hi - q.macro_f1], capsize=2)
        for xi, val in zip(x, q.macro_f1):
            ax.text(xi, 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=90, color="white")
    ax.set_xticks(range(len(views)), [cfg.cam_label(v) for v in views])
    ax.set_ylabel("Makro F1 (OOF, %95 GA)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    _savefig(fig, "F3_macroF1_by_view", cfg)
    plt.close(fig)

    # F4 karışıklık matrisleri (satır normalize)
    fig, axes = plt.subplots(len(models), len(views), figsize=(2.3 * len(views), 2.3 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, v in enumerate(views):
            ax = axes[i, j]
            if (m, v) not in oof:
                ax.axis("off")
                continue
            M = plant_cms[(m, v)].sum(0).astype(float)
            M = M / np.maximum(M.sum(1, keepdims=True), 1)
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
            for a in range(K):
                for b in range(K):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=5,
                            color="white" if M[a, b] > 0.5 else "black")
            ticks = [CLASS_SHORT[c] for c in CLASSES]
            ax.set_xticks(range(K), ticks, fontsize=6)
            ax.set_yticks(range(K), ticks, fontsize=6)
            ax.set_title(f"{m}\n{cfg.cam_label(v)}", fontsize=7)
            if j == 0:
                ax.set_ylabel("Gerçek")
            if i == len(models) - 1:
                ax.set_xlabel("Tahmin")
    fig.tight_layout()
    _savefig(fig, "F4_confusion_matrices", cfg)
    plt.close(fig)

    # F5 sınıf x görünüm F1 ısı haritası
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 2.6), squeeze=False)
    for i, m in enumerate(models):
        H = t4[t4.model == m].pivot(index="sinif", columns="view", values="f1")
        H = H.reindex(index=[CLASS_SHORT[c] for c in CLASSES], columns=[cfg.cam_label(v) for v in views])
        ax = axes[0, i]
        ax.imshow(H.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for a in range(H.shape[0]):
            for b in range(H.shape[1]):
                ax.text(b, a, f"{H.values[a, b]:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if H.values[a, b] > 0.6 else "white")
        ax.set_xticks(range(H.shape[1]), H.columns, fontsize=7)
        ax.set_yticks(range(H.shape[0]), H.index, fontsize=7)
        ax.set_title(m, fontsize=8)
    _savefig(fig, "F5_per_class_f1_heatmap", cfg)
    plt.close(fig)

    # F6 kamera çifti farkları (orman grafiği)
    if not t5.empty:
        fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(t5) + 0.8))
        y = np.arange(len(t5))[::-1]
        dcol = "d_macro_f1(B-A)"
        ax.errorbar(t5[dcol], y, xerr=[t5[dcol] - t5.d_macro_f1_ci_lo, t5.d_macro_f1_ci_hi - t5[dcol]],
                    fmt="o", color="#333333", ms=3, capsize=2)
        for yi, hi, sig in zip(y, t5.d_macro_f1_ci_hi, t5["significant_0.05"]):
            if sig:
                ax.text(hi, yi, " *", va="center", fontsize=9)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(y, [f"{b.split(' ', 1)[1]} − {a.split(' ', 1)[1]}  ({a.split(' ')[0]})" for a, b in zip(t5.A, t5.B)], fontsize=6.5)
        ax.set_xlabel("Makro F1 farkı (%95 GA, bitki kümeli bootstrap); * Holm p<0,05")
        _savefig(fig, "F6_camera_pair_differences", cfg)
        plt.close(fig)

    # F7 maliyet–başarı
    if "T7b_cost_accuracy_tradeoff" in tables and (T / "T7_cost.csv").exists():
        cost = tables["T7_cost"]
        dev = "gpu" if "gpu_single_ms_median" in cost.columns else "cpu"
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        for i, m in enumerate(models):
            c = cost[cost.model == m]
            if c.empty:
                continue
            q = t3[t3.model == m]
            xs = [c.iloc[0][f"{dev}_single_ms_median"] if v != FUSION else c.iloc[0][f"{dev}_fusion_ms_median"] for v in q.view_key]
            ax.scatter(xs, q.macro_f1, color=colors[i], label=m, s=18)
            for x, yv, lab in zip(xs, q["macro_f1"], q["view"]):
                ax.annotate(lab, (x, yv), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"Oturum başına gecikme ({dev.upper()}, ms, medyan)")
        ax.set_ylabel("Makro F1 (OOF)")
        ax.legend(frameon=False, fontsize=7)
        _savefig(fig, "F7_cost_vs_accuracy", cfg)
        plt.close(fig)

    # F8 eğitim eğrileri (kat 0)
    try:
        fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 2.5), squeeze=False, sharey=True)
        for i, m in enumerate(models):
            ax = axes[0, i]
            for j, c in enumerate(cams):
                f = d["runs"] / f"{m}_cam{c}_fold0" / "results.csv"
                if f.exists():
                    h = pd.read_csv(f)
                    h.columns = [x.strip() for x in h.columns]
                    col = [x for x in h.columns if "accuracy_top1" in x]
                    if col:
                        ax.plot(h["epoch"], h[col[0]], color=colors[j], lw=1, label=cfg.cam_label(c))
            ax.set_title(f"{m} (kat 0)", fontsize=8)
            ax.set_xlabel("Epok")
        axes[0, 0].set_ylabel("Doğrulama doğruluğu")
        axes[0, 0].legend(frameon=False, fontsize=6)
        _savefig(fig, "F8_training_curves_fold0", cfg)
        plt.close(fig)
    except Exception as e:
        LOG.warning("F8 üretilemedi: %s", e)

    _write_xlsx(tables, cfg)
    summary = {
        "design": {"n_sessions": int(len(man)), "n_plants": int(man.plant.nunique()), "cameras": cams,
                   "camera_labels": {str(k): v for k, v in cfg.camera_labels.items()},
                   "n_folds": cfg.n_folds, "train_seed": cfg.train_seed, "pose_seed": cfg.pose_seed,
                   "fold_random_state": int(_read(d["manifests"] / "plant_folds.csv").random_state.iloc[0]),
                   "imgsz": cfg.imgsz, "epochs": cfg.epochs, "patience": cfg.patience, "batch": cfg.batch,
                   "bootstrap_B": cfg.bootstrap_B, "models": list(models)},
        "metadata_audit": json.loads((T / "T0_metadata_audit.json").read_text()) if (T / "T0_metadata_audit.json").exists() else None,
        "tables": {k: v.to_dict("records") for k, v in tables.items()},
    }
    try:
        summary["environment"] = json.loads((d["logs"] / "environment.json").read_text())
    except Exception:
        pass
    try:
        (cfg.D / "results" / "SUMMARY_FOR_MANUSCRIPT.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    except Exception as e:
        LOG.error("Özet JSON yazılamadı (%s). Tablolar ve şekiller yazıldı; "
                  "makale için ALL_TABLES.xlsx kullanılabilir.", e)
        raise
    LOG.info("Analiz tamamlandı.")


def _json_default(o):
    """NumPy/pandas türlerini JSON'a çevirir; çevrilemeyeni metne düşürür."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


def _write_xlsx(tables: dict, cfg: Config):
    T = cfg.dirs()["tables"]
    with pd.ExcelWriter(T / "ALL_TABLES.xlsx", engine="openpyxl") as xl:
        for name, df in tables.items():
            _atomic_csv(df, T / f"{name}.csv")
            df.to_excel(xl, sheet_name=name[:31], index=False)
