
"""compare_pretrained_checkpoints.py

Makalenin "Modeller ve Eğitim" bölümündeki mimari karşılaştırmasını yeniden üretir.

Ne yapar:
  1. Ultralytics üzerinden yolo11n-cls.pt ve yolo26n-cls.pt kontrol noktalarını yükler.
  2. Her ikisinin SHA-256 özetini, dosya boyutunu ve parametre sayısını yazdırır
     (makaledeki Ek Tablo S3'ün kaynağı).
  3. state_dict sözlüklerini karşılaştırır: tensör adlarının ve boyutlarının eşleşip
     eşleşmediğini ve ortak tensörlerdeki ortalama mutlak ağırlık farkını hesaplar.

Kullanım:
    python scripts/compare_pretrained_checkpoints.py

Beklenen çıktı (Ultralytics 8.4.155, 18.09.2026 tarihli kontrol noktaları):
    yolo11n-cls.pt  sha256=c62d41bf...02bd7  bayt=5790624  parametre=2812104
    yolo26n-cls.pt  sha256=0dd6f8db...79e81  bayt=5786434  parametre=2812104
    tensör sayısı: 236 / 236 | ad ve boyut eşleşmesi: True
    ortak tensörlerde ortalama mutlak fark: 2.064456
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from ultralytics import YOLO

MODELS = ("yolo11n-cls.pt", "yolo26n-cls.pt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    states, paths = {}, {}
    for name in MODELS:
        model = YOLO(name)                      # yoksa indirir
        path = Path(model.ckpt_path if hasattr(model, "ckpt_path") else name)
        if not path.exists():                   # Ultralytics önbelleğinde arar
            cand = list(Path.home().rglob(name)) + list(Path(".").rglob(name))
            path = cand[0] if cand else path
        n_par = sum(p.numel() for p in model.model.parameters())
        print(f"{name}  sha256={sha256(path) if path.exists() else 'dosya bulunamadı'}"
              f"  bayt={path.stat().st_size if path.exists() else '-'}  parametre={n_par}")
        states[name] = model.model.state_dict()
        paths[name] = path

    a, b = states[MODELS[0]], states[MODELS[1]]
    same_names = list(a.keys()) == list(b.keys())
    shared = [k for k in a if k in b and a[k].shape == b[k].shape]
    print(f"tensör sayısı: {len(a)} / {len(b)} | ad ve boyut eşleşmesi: {same_names and len(shared) == len(a)}")

    if shared:
        diff = torch.stack([(a[k].float() - b[k].float()).abs().mean() for k in shared]).mean()
        print(f"ortak tensörlerde ortalama mutlak fark: {float(diff):.6f}")


if __name__ == "__main__":
    main()
