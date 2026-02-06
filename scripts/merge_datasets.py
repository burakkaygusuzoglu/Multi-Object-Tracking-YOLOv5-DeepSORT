from pathlib import Path
import shutil

# --- KAYNAK PATHLER ---
ROBO = Path(r"C:\Users\Burak\OneDrive\Desktop\roboflow person-ball")  # temizlenmiş Roboflow (sadece person=0)
KAG   = Path(r"C:\Users\Burak\OneDrive\Desktop\kaggle_person_dataset")

# --- HEDEF (YOLOv5 içinde) ---
OUT = Path(r"C:\Users\Burak\yolov5\dataset_merged")

# Kaggle split isimleri
KAG_TRAIN = KAG / "images" / "train"
KAG_VAL   = KAG / "images" / "validate"   # bazı datasetlerde "validate" olur
KAG_TEST  = KAG / "images" / "test"       # varsa
KAG_LTRAIN = KAG / "labels" / "train"
KAG_LVAL   = KAG / "labels" / "validate"
KAG_LTEST  = KAG / "labels" / "test"

# Roboflow split isimleri
ROBO_TRAIN = ROBO / "train" / "images"
ROBO_VAL   = ROBO / "valid" / "images"
ROBO_LTRAIN = ROBO / "train" / "labels"
ROBO_LVAL   = ROBO / "valid" / "labels"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def ensure_dirs():
    for p in [
        OUT/"images/train", OUT/"images/val",
        OUT/"labels/train", OUT/"labels/val"
    ]:
        p.mkdir(parents=True, exist_ok=True)

def copy_split(img_dir, lbl_dir, out_split, prefix):
    """img_dir + lbl_dir içindeki eşleşen image/label çiftlerini OUT'a kopyalar, isim çakışmasını prefix ile çözer."""
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"SKIP (yok): {img_dir}  |  {lbl_dir}")
        return 0

    out_img = OUT / "images" / out_split
    out_lbl = OUT / "labels" / out_split

    n = 0
    for img in img_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue

        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue

        new_stem = f"{prefix}_{img.stem}"
        new_img = out_img / (new_stem + img.suffix.lower())
        new_lbl = out_lbl / (new_stem + ".txt")

        shutil.copy2(img, new_img)
        shutil.copy2(lbl, new_lbl)
        n += 1

    print(f"OK: {prefix} {out_split} -> {n} pair kopyalandı")
    return n

def main():
    ensure_dirs()

    total = 0
    # Roboflow -> train/val
    total += copy_split(ROBO_TRAIN, ROBO_LTRAIN, "train", "robo")
    total += copy_split(ROBO_VAL,   ROBO_LVAL,   "val",   "robo")

    # Kaggle -> train/val (validate varsa onu val yapıyoruz)
    total += copy_split(KAG_TRAIN, KAG_LTRAIN, "train", "kag")
    if KAG_VAL.exists() and KAG_LVAL.exists():
        total += copy_split(KAG_VAL, KAG_LVAL, "val", "kag")
    elif (KAG / "images" / "valid").exists() and (KAG / "labels" / "valid").exists():
        total += copy_split(KAG / "images" / "valid", KAG / "labels" / "valid", "val", "kag")
    else:
        print("Uyarı: Kaggle val/validate bulunamadı. Sadece train kopyalandı.")

    print(f"\nBİTTİ. Toplam kopyalanan pair: {total}")
    print(f"Çıktı klasörü: {OUT}")

if __name__ == "__main__":
    main()
