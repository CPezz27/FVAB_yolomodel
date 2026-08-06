import cv2
import albumentations as A
import os
import random
import shutil
from pathlib import Path
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec



INPUT_IMAGES_DIR = r"C:\Users\divic\Desktop\utiliGTSDB\images"
INPUT_LABELS_DIR = r"C:\Users\divic\Desktop\utiliGTSDB\output_labels"
OUTPUT_IMAGES_DIR = r"C:\Users\divic\Desktop\utiliGTSDB\images_augumented"
OUTPUT_LABELS_DIR = r"C:\Users\divic\Desktop\utiliGTSDB\output_labels_augumented"

AUGMENTATIONS_PER_IMAGE = 6 

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

REPORT_DIR = Path("reportsAg")
REPORT_DIR.mkdir(exist_ok=True)
labels_path = r'C:\Users\divic\Desktop\utiliGTSDB\labels'
output_path = r'C:\Users\divic\Desktop\utiliGTSDB\output_labels'

CLASS_NAMES = {
    0: "stop_signal", 1: "limit10", 2: "limit20", 3: "forward_right",     
    4: "green_light", 5: "yellow_light", 6: "red_light", 7: "pedon",             
    8: "front_car", 9: "bottom_car", 10: "left_car", 11: "right_car"
}

CLASS_MAPPING_RULES = {40: 12, 0: 2, 14: 0, 33: 13, 34: 14, 36: 15, 37: 16}


def visualizza_yolo_bboxes(file_path, canvas_width=1000, canvas_height=1000):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(canvas_height, 0)
    ax.set_aspect('equal')
    ax.set_title(f"Visualizzazione Bounding Boxes: {file_path}")

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            if not parts: continue
            
            
            cls, x_center, y_center, width, height = map(float, parts)

            w = width * canvas_width
            h = height * canvas_height
            x = (x_center * canvas_width) - (w / 2)
            y = (y_center * canvas_height) - (h / 2)

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            plt.text(x, y - 5, f"Cls: {int(cls)}", color='red', fontsize=8, fontweight='bold')

        plt.show()

    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


def parse_labels(label_dir: Path) -> dict:
    """
    Legge tutti i .txt in label_dir.
    Ritorna {filename: [(class_id, cx, cy, w, h), ...]}
    """
    data = {}
    for txt in sorted(label_dir.glob("*.txt")):
        boxes = []
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    boxes.append((cls, cx, cy, w, h))
        data[txt.stem] = boxes
    return data
 

def compute_stats(data: dict) -> dict:
    class_counts   = Counter()
    bbox_areas     = defaultdict(list)
    bbox_widths    = defaultdict(list)
    bbox_heights   = defaultdict(list)
    cx_positions   = defaultdict(list)
    cy_positions   = defaultdict(list)
    boxes_per_img  = []
    empty_images   = []
 
    for img_name, boxes in data.items():
        if not boxes:
            empty_images.append(img_name)
        boxes_per_img.append(len(boxes))
        for cls, cx, cy, w, h in boxes:
            class_counts[cls] += 1
            bbox_areas[cls].append(w * h)
            bbox_widths[cls].append(w)
            bbox_heights[cls].append(h)
            cx_positions[cls].append(cx)
            cy_positions[cls].append(cy)
 
    total_boxes  = sum(class_counts.values())
    total_images = len(data)
 
    # Individua classi rare (< 5% del totale o < 50 istanze)
    minority = {
        c: n for c, n in class_counts.items()
        if n < max(50, 0.05 * total_boxes)
    }
 
    return {
        "total_images":    total_images,
        "total_boxes":     total_boxes,
        "class_counts":    dict(class_counts),
        "bbox_areas":      dict(bbox_areas),
        "bbox_widths":     dict(bbox_widths),
        "bbox_heights":    dict(bbox_heights),
        "cx_positions":    dict(cx_positions),
        "cy_positions":    dict(cy_positions),
        "boxes_per_img":   boxes_per_img,
        "empty_images":    empty_images,
        "minority_classes": minority,
    }
 
def class_label(cls: int) -> str:
    return CLASS_NAMES.get(cls, f"cls_{cls}")
 
 
def plot_class_distribution(stats: dict):
    counts = stats["class_counts"]
    if not counts:
        return
    classes = sorted(counts.keys())
    labels  = [class_label(c) for c in classes]
    values  = [counts[c] for c in classes]
    total   = sum(values)
    colors  = ["#e74c3c" if c in stats["minority_classes"] else "#3498db" for c in classes]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribuzione classi nel dataset", fontsize=14, fontweight="bold")
 
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Conteggio istanze per classe\n(rosso = classe minoritaria)")
    axes[0].set_ylabel("Numero di bbox")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{v}\n({v/total*100:.1f}%)", ha="center", va="bottom", fontsize=8)
 
    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=140, pctdistance=0.8)
    axes[1].set_title("Proporzioni per classe")
 
    plt.tight_layout()
    path = REPORT_DIR / "01_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")
 
 
def plot_bbox_size_analysis(stats: dict):
    areas  = stats["bbox_areas"]
    if not areas:
        return
 
    classes = sorted(areas.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analisi dimensioni bounding box", fontsize=14, fontweight="bold")
 
    ax = axes[0, 0]
    for cls in classes:
        a = areas[cls]
        ax.scatter([class_label(cls)] * len(a), a, alpha=0.3, s=10)
    ax.set_title("Area bbox (w×h normalizzata) per classe")
    ax.set_ylabel("Area normalizzata")
    ax.tick_params(axis="x", rotation=30)
 
    ax = axes[0, 1]
    all_areas = [a for lst in areas.values() for a in lst]
    ax.hist(all_areas, bins=50, color="#3498db", edgecolor="white", linewidth=0.3)
    ax.axvline(0.0256, color="orange", linestyle="--", label="small (< 32²px in 640)")
    ax.axvline(0.1024, color="green",  linestyle="--", label="medium")
    ax.set_title("Distribuzione aree (global)")
    ax.set_xlabel("Area normalizzata")
    ax.legend(fontsize=8)
 
    ax = axes[1, 0]
    for cls in classes:
        ws = stats["bbox_widths"][cls]
        hs = stats["bbox_heights"][cls]
        ar = [w / h if h > 0 else 0 for w, h in zip(ws, hs)]
        ax.violinplot([ar], positions=[cls], showmedians=True)
    ax.set_title("Aspect ratio (w/h) per classe")
    ax.set_ylabel("w / h")
    ax.set_xticks(classes)
    ax.set_xticklabels([class_label(c) for c in classes], rotation=30)
 
    ax = axes[1, 1]
    all_cx = [x for lst in stats["cx_positions"].values() for x in lst]
    all_cy = [y for lst in stats["cy_positions"].values() for y in lst]
    h2d, _, _ = np.histogram2d(all_cx, all_cy, bins=20, range=[[0,1],[0,1]])
    im = ax.imshow(h2d.T, origin="lower", cmap="hot", extent=[0,1,0,1])
    plt.colorbar(im, ax=ax, label="densità")
    ax.set_title("Heatmap centroidi (dove si concentrano gli oggetti)")
    ax.set_xlabel("cx (sinistra → destra)")
    ax.set_ylabel("cy (alto → basso)")
 
    plt.tight_layout()
    path = REPORT_DIR / "02_bbox_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")
 
 
def plot_images_per_class(stats: dict):
    bpi = stats["boxes_per_img"]
    if not bpi:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Analisi immagini", fontsize=14, fontweight="bold")
 
    axes[0].hist(bpi, bins=range(0, max(bpi) + 2), color="#2ecc71", edgecolor="white")
    axes[0].set_title("Numero di bbox per immagine")
    axes[0].set_xlabel("Bbox per immagine")
    axes[0].set_ylabel("Numero immagini")
 
    labels_ = ["Con oggetti", "Vuote (solo background)"]
    n_empty = len(stats["empty_images"])
    sizes_  = [stats["total_images"] - n_empty, n_empty]
    colors_ = ["#3498db", "#e74c3c"]
    axes[1].pie(sizes_, labels=labels_, colors=colors_, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Immagini con vs senza annotazioni")
 
    plt.tight_layout()
    path = REPORT_DIR / "03_images_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")
 
 
def plot_cooccurrence(data: dict, stats: dict):
    """Matrice di co-occorrenza: quali classi appaiono insieme?"""
    classes = sorted(stats["class_counts"].keys())
    n = len(classes)
    if n < 2:
        return
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
 
    for boxes in data.values():
        cls_in_img = set(b[0] for b in boxes)
        for c1 in cls_in_img:
            for c2 in cls_in_img:
                if c1 in idx and c2 in idx:
                    matrix[idx[c1], idx[c2]] += 1
 
    labels = [class_label(c) for c in classes]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label="co-occorrenze")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=9, color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
    ax.set_title("Co-occorrenza classi\n(quali oggetti appaiono nella stessa immagine)")
    plt.tight_layout()
    path = REPORT_DIR / "04_cooccurrence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")
 

def generate_report(stats: dict) -> dict:
    total   = stats["total_boxes"]
    counts  = stats["class_counts"]
 
    recommendations = []
 
    if stats["minority_classes"]:
        minority_str = ", ".join(
            f"{class_label(c)} ({n} istanze, {n/total*100:.1f}%)"
            for c, n in stats["minority_classes"].items()
        )
        recommendations.append({
            "severity": "HIGH",
            "issue": "Classi minoritarie rilevate",
            "details": minority_str,
            "action": (
                "1. Raccolta dati mirata per queste classi. "
                "2. Usa copy-paste augmentation (CutPaste/MosaicMix). "
                "3. Aumenta il peso della loss su queste classi con cls_pw in YOLOv8/v11. "
                "4. Considera oversampling a livello di immagine."
            )
        })
 
    n_empty = len(stats["empty_images"])
    if n_empty > 0:
        pct = n_empty / stats["total_images"] * 100
        recommendations.append({
            "severity": "MEDIUM",
            "issue": f"{n_empty} immagini senza annotazioni ({pct:.1f}%)",
            "details": f"Immagini: {stats['empty_images'][:5]}{'...' if n_empty > 5 else ''}",
            "action": (
                "Verificare se sono background samples intenzionali "
                "(utile per ridurre falsi positivi in pista). "
                "In YOLO v8/v11 basta lasciarli vuoti — vengono gestiti automaticamente."
            )
        })
 
    all_areas = [a for lst in stats["bbox_areas"].values() for a in lst]
    if all_areas:
        pct_small = sum(1 for a in all_areas if a < 0.0256) / len(all_areas) * 100
        if pct_small > 30:
            recommendations.append({
                "severity": "HIGH",
                "issue": f"Molte bbox piccole: {pct_small:.1f}% < 32×32px equivalente",
                "details": "Su Jetson Nano / pista reale, gli oggetti piccoli sono critici",
                "action": (
                    "1. Usa imgsz=1280 in training (non 640). "
                    "2. Applica SAHI (Slicing Aided Hyper Inference) in inferenza. "
                    "3. Considera multi-scale training (scales: [0.5, 1.5])."
                )
            })
 
    recommendations.append({
        "severity": "HIGH",
        "issue": "Gap train-accuracy vs on-track accuracy",
        "details": "Problema comune: domain shift tra foto statiche e video da auto in movimento",
        "action": (
            "1. Augmentation dinamica: motion blur (kernel 15-31px, angolo 0-360°). "
            "2. Aggiungi immagini catturate DALLA macchina (stessa angolazione e altezza camera). "
            "3. Normalizza input con statistiche del tuo dataset (non ImageNet). "
            "4. Usa mosaic=1.0 + mixup=0.1 in YOLO config. "
            "5. Testa con confidence threshold più basso (0.25 invece di 0.5). "
            "6. Verifica che train/val split sia estratto da sequenze diverse (no data leakage da stesso video)."
        )
    })
 
    report = {
        "summary": {
            "total_images":   stats["total_images"],
            "total_boxes":    stats["total_boxes"],
            "num_classes":    len(counts),
            "avg_boxes_per_image": round(np.mean(stats["boxes_per_img"]), 2),
            "empty_images":   n_empty,
        },
        "class_distribution": {
            class_label(c): {"count": n, "pct": round(n / total * 100, 2)}
            for c, n in sorted(counts.items())
        },
        "minority_classes": {
            class_label(c): n for c, n in stats["minority_classes"].items()
        },
        "recommendations": recommendations
    }
    return report
 
 
def print_summary(report: dict):
    s = report["summary"]
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)
    print(f"  Immagini totali      : {s['total_images']}")
    print(f"  Bounding box totali  : {s['total_boxes']}")
    print(f"  Classi               : {s['num_classes']}")
    print(f"  Media bbox/immagine  : {s['avg_boxes_per_image']}")
    print(f"  Immagini vuote       : {s['empty_images']}")
    print("\n  DISTRIBUZIONE CLASSI")
    print("  " + "-"*40)
    for name, info in report["class_distribution"].items():
        bar = "|" * int(info["pct"] / 2)
        flag = "MINORITARIA" if name in report.get("minority_classes", {}) else ""
        print(f"  {name:<20} {info['count']:>5}  {info['pct']:>5.1f}%  {bar}{flag}")
    print("\n  RACCOMANDAZIONI")
    print("  " + "-"*40)
    for r in report["recommendations"]:
        sev = {"HIGH": "RED", "MEDIUM": "YELLOW", "LOW": "GREEN"}.get(r["severity"], "WHITE")
        print(f"\n  {sev} [{r['severity']}] {r['issue']}")
        print(f"     Azione: {r['action'][:120]}...")
    print("\n" + "="*55)
 
 
def report_dataset():
    parser = argparse.ArgumentParser(description="Dataset analysis for YOLO OD")
    parser.add_argument("--labels",     type=str, default=r"C:\Users\divic\Desktop\FVAB\DatasetAgumented\labels",
                        help="Cartella con i file .txt delle label (YOLO format)")
    parser.add_argument("--images",     type=str, default=r"C:\Users\divic\Desktop\FVAB\DatasetAgumented\images",
                        help="Cartella immagini (opzionale, per verifica corrispondenza)")
    args = parser.parse_args()
 
    label_dir = Path(args.labels)
    if not label_dir.exists():
        print(f"ERRORE: cartella non trovata: {label_dir}")
        return
 
    print(f"\n Analisi dataset: {label_dir}")
    print(f" Output:          {REPORT_DIR}/\n")
 
    data  = parse_labels(label_dir)
    stats = compute_stats(data)
 
    if not data:
        print("Nessun file .txt trovato. Controlla il percorso.")
        return
 
    if args.images:
        img_dir = Path(args.images)
        img_stems  = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        lbl_stems  = set(data.keys())
        only_imgs  = img_stems - lbl_stems
        only_lbls  = lbl_stems - img_stems
        if only_imgs:
            print(f"  ATTENZIONE  {len(only_imgs)} immagini senza label: {list(only_imgs)}")
        if only_lbls:
            print(f"  ATTENZIONE  {len(only_lbls)} label senza immagine: {list(only_lbls)}")
 
    print("  Generazione grafici...")
    plot_class_distribution(stats)
    plot_bbox_size_analysis(stats)
    plot_images_per_class(stats)
    plot_cooccurrence(data, stats)
 
    report = generate_report(stats)
    report_path = REPORT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Salvato: {report_path}")
 
    print_summary(report)


def load_labels(label_path):
    labels = []
    label_path = Path(label_path)
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0])] + [float(v) for v in parts[1:]])
    return labels


def save_labels(label_path, labels):
    with open(label_path, "w") as f:
        for lb in labels:
            cls = int(lb[0])
            xc, yc, w, h = lb[1], lb[2], lb[3], lb[4]
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def clip_labels(labels):
    """
    Conversione corretta:
      1) Da [xc, yc, w, h] → [x1, y1, x2, y2]
      2) Clipping in [0, 1]
      3) Ritorno a [xc, yc, w, h]
    Evita il bug della formula precedente che azzerava w/h vicino ai bordi.
    """
    valid = []
    for lb in labels:
        cls, xc, yc, w, h = lb

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        x1 = float(np.clip(x1, 0.0, 1.0))
        y1 = float(np.clip(y1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))
        y2 = float(np.clip(y2, 0.0, 1.0))

        new_w = x2 - x1
        new_h = y2 - y1

        if new_w > 1e-4 and new_h > 1e-4:
            new_xc = x1 + new_w / 2
            new_yc = y1 + new_h / 2
            valid.append([cls, new_xc, new_yc, new_w, new_h])

    return valid


def random_translation(img, labels, max_shift=0.3):
    if img is None:
        return None, labels

    h, w = img.shape[:2]
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114)
    )

    tx_norm = tx / w
    ty_norm = ty / h

    new_labels = []
    for lb in labels:
        cls, xc, yc, bw, bh = lb
        new_labels.append([cls, xc + tx_norm, yc + ty_norm, bw, bh])

    return translated_img, clip_labels(new_labels)

def load_mosaic(img_paths, label_dir, output_size=640):
    if len(img_paths) < 4:
        return None, None

    s = output_size
    label_dir = Path(label_dir)

    mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    mosaic_labels = []

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    indices = random.sample(range(len(img_paths)), 4)

    for i, idx in enumerate(indices):
        img_p = img_paths[idx]
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        img = cv2.resize(img, (s, s))

        row, col = positions[i]
        mosaic_img[row * s:(row + 1) * s, col * s:(col + 1) * s] = img

        label_path = label_dir / (img_p.stem + ".txt")
        labels = load_labels(label_path)

        for lb in labels:
            cls, xc, yc, bw, bh = lb
            new_xc = (xc + col) / 2
            new_yc = (yc + row) / 2
            new_bw = bw / 2
            new_bh = bh / 2
            mosaic_labels.append([cls, new_xc, new_yc, new_bw, new_bh])

    final_img = cv2.resize(mosaic_img, (s, s))
    return final_img, clip_labels(mosaic_labels)


def process_images(img_in, img_out, lbl_in, lbl_out, n_translations=500, n_mosaics=500):
    img_in, img_out = Path(img_in), Path(img_out)
    lbl_in, lbl_out = Path(lbl_in), Path(lbl_out)
    
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    img_list = [p for p in img_in.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not img_list: return print("Nessuna immagine trovata!")

    print("Copia file originali...")
    for img_p in img_list:
        shutil.copy2(img_p, img_out / img_p.name)
        l_p = lbl_in / f"{img_p.stem}.txt"
        if l_p.exists():
            shutil.copy2(l_p, lbl_out / l_p.name)

    print(f"Generazione di {n_translations} traslazioni...")
    for i in range(n_translations):
        img_p = random.choice(img_list)
        img = cv2.imread(str(img_p))
        labels = load_labels(lbl_in / f"{img_p.stem}.txt")
        
        trans_img, trans_labels = random_translation(img, labels)
        out_name = f"trans_{i}_{img_p.stem}"
        cv2.imwrite(str(img_out / f"{out_name}.jpg"), trans_img)
        save_labels(lbl_out / f"{out_name}.txt", trans_labels)

    print(f"Generazione di {n_mosaics} mosaici...")
    for i in range(n_mosaics):
        m_img, m_lbl = load_mosaic(img_list, lbl_in)
        if m_img is not None:
            out_name = f"mosaic_{i}"
            cv2.imwrite(str(img_out / f"{out_name}.jpg"), m_img)
            save_labels(lbl_out / f"{out_name}.txt", m_lbl)

    print("Completato!")

'''
ADDESTRAMENTO MODELLO MODO VECCHIO
class YOLODatasetSplitter:
    def __init__(self, img_dir, lbl_dir, output_dir, val_ratio=0.2, random_seed=42):
        self.img_dir_all = img_dir
        self.lbl_dir_all = lbl_dir
        self.dataset_dir = output_dir  # Dove verranno creati i file per YOLO
        self.val_ratio = val_ratio
        random.seed(random_seed)
        
        # Directory di output strutturate per YOLO
        self.img_dir_train = os.path.join(self.dataset_dir, 'images', 'train')
        self.img_dir_val = os.path.join(self.dataset_dir, 'images', 'val')
        self.lbl_dir_train = os.path.join(self.dataset_dir, 'labels', 'train')
        self.lbl_dir_val = os.path.join(self.dataset_dir, 'labels', 'val')
        
        self.class_to_images = defaultdict(set)
        self.image_to_classes = {}
        self.val_counts = defaultdict(int)
        
        self._create_directories()

    def _create_directories(self):
        for d in [self.img_dir_train, self.img_dir_val, 
                  self.lbl_dir_train, self.lbl_dir_val]:
            os.makedirs(d, exist_ok=True)

    # Modifica shutil.move in shutil.copy per sicurezza
    def _copy_files(self, img_list, dest_img_dir, dest_lbl_dir):
        for img_path in img_list:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            lbl_path = os.path.join(self.lbl_dir_all, f"{base_name}.txt")
            
            shutil.copy(img_path, os.path.join(dest_img_dir, filename))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(dest_lbl_dir, f"{base_name}.txt"))
    
    def _map_classes_to_images(self):
        image_paths = sorted(glob(os.path.join(self.img_dir_all, '*')))
        image_paths = [f for f in image_paths if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')]
        
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(self.lbl_dir_all, f"{base_name}.txt")
            
            if not os.path.exists(lbl_path):
                continue
                
            with open(lbl_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            classes = set()
            for line in lines:
                try:
                    class_id = int(line.split()[0])
                    classes.add(class_id)
                    self.class_to_images[class_id].add(img_path)
                except (ValueError, IndexError):
                    continue
            
            if classes:
                self.image_to_classes[img_path] = classes
        
        self.val_targets = {cls: max(1, int(len(imgs) * self.val_ratio)) 
                          for cls, imgs in self.class_to_images.items()}
    
    def _split_images(self):
        val_images = set()
        all_images = list(self.image_to_classes.keys())
        random.shuffle(all_images)
        
        for img_path in all_images:
            classes = self.image_to_classes[img_path]
            should_go_to_val = any(self.val_counts[cls] < self.val_targets[cls] for cls in classes)
            
            if should_go_to_val:
                val_images.add(img_path)
                for cls in classes:
                    if self.val_counts[cls] < self.val_targets[cls]:
                        self.val_counts[cls] += 1
        
        train_images = [img for img in all_images if img not in val_images]
        return train_images, list(val_images)

    def create_yolo_yaml(self):
        if not self.class_to_images:
            raise ValueError("ERRORE: Nessuna classe trovata! Controlla i file .txt nelle labels.")
        
        names_dict = {int(i): f'Class_{i}' for i in self.class_to_images.keys()}
        
        data_yaml = {
            'path': os.path.abspath(self.dataset_dir), # Usa percorsi assoluti per sicurezza
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(names_dict),
            'names': names_dict
        }

        # Definiamo dove salvare il file .yaml
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        
        # Scriviamo il file su disco
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        print(f"✅ File configurazione creato in: {yaml_path}")
        return yaml_path  # <--- FONDAMENTALE: restituisci il percorso!
    

    def split_dataset(self):
        print(f"DEBUG: Cerco immagini in: {self.img_dir_all}")
        file_presenti = glob(os.path.join(self.img_dir_all, '*'))
        print(f"DEBUG: File trovati nella cartella immagini: {len(file_presenti)}")
        print("⏳ Mappatura classi...")
        self._map_classes_to_images()
        print("⏳ Suddivisione train/val...")
        train_images, val_images = self._split_images()
        print("⏳ Spostamento file...")
        self._copy_files(train_images, self.img_dir_train, self.lbl_dir_train)
        self._copy_files(val_images, self.img_dir_val, self.lbl_dir_val)
        
        print(f"\n✅ Split completato! Train: {len(train_images)}, Val: {len(val_images)}")
        return self.create_yolo_yaml()

# --- ESECUZIONE ---

if __name__ == "__main__":
    PATH_IMMAGINI = r"C:\Users\divic\Desktop\Test\ImmaginiTest\ImmaginiTest\ImmaginiAumentate"
    PATH_LABEL = r"C:\Users\divic\Desktop\Test\ImmaginiTest\LabelTest\LabelAumentate"
    
    # Percorso dove vuoi creare il dataset strutturato per YOLO
    # Ti consiglio di crearne uno nuovo per non fare confusione
    PATH_OUTPUT = r"C:\Users\divic\Desktop\Test\Dataset_YOLO"

    # Inizializziamo lo splitter (dovremo modificare leggermente la classe sotto)
    splitter = YOLODatasetSplitter(img_dir=PATH_IMMAGINI, lbl_dir=PATH_LABEL, output_dir=PATH_OUTPUT)
    yaml_config_path = splitter.split_dataset()
    
    # ... resto del training

    # 3. Training
    model = YOLO("yolo11n.pt")
    
    print("\n🚀 Avvio Training...")
    results = model.train(
        data=yaml_config_path,  # Passiamo il file YAML creato, non la cartella!
        epochs=50,
        patience=5,
        imgsz=360,
        batch=64,
        optimizer="AdamW",
        lr0=2e-4,
        lrf=1e-2,
        weight_decay=0.01,
        warmup_epochs=3,
        rect=False,
        augment=False,
        deterministic=True,
        seed=0,
    )

    # 4. Visualizzazione Metriche principali
    print("\n📊 RIEPILOGO METRICHE:")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")
'''

class_mapping = {
    40: 12,
    0: 2,
    14: 0,
    33: 13,
    34: 14,
    36: 15,
    37: 16
}

if not os.path.exists(output_path):
    os.makedirs(output_path)

def remap_labels():
    files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    for filename in files:
        with open(os.path.join(labels_path, filename), 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            old_cls = int(parts[0])
            
            # Applica il remapping se la classe è nel dizionario
            if old_cls in class_mapping:
                parts[0] = str(class_mapping[old_cls])
                new_lines.append(" ".join(parts) + "\n")
            else:
                # Opzionale: decidi cosa fare con le classi non mappate
                # Per ora le manteniamo così come sono
                new_lines.append(line)

        # Scrittura del nuovo file
        with open(os.path.join(output_path, filename), 'w') as f:
            f.writelines(new_lines)

    print(f"Processo completato! I file rimappati sono in: {output_path}")

def process_dataset():

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, 
                        border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.6),
        
        A.Perspective(scale=(0.02, 0.08), p=0.4)
        
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

    images = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(INPUT_IMAGES_DIR, img_name)
        txt_path = os.path.join(INPUT_LABELS_DIR, f"{base_name}.txt")
        
        if not os.path.exists(txt_path):
            print(f"Attenzione: Label mancante per {img_name}. Salto.")
            continue
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes = []
        class_labels = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
        
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, img_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt"), 'w') as f:
            with open(txt_path, 'r') as orig_f:
                f.write(orig_f.read())
        
        for i in range(AUGMENTATIONS_PER_IMAGE):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                t_img = transformed['image']
                t_bboxes = transformed['bboxes']
                t_classes = transformed['class_labels']
                
                new_base_name = f"{base_name}_aug_{i}"
                
                cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{new_base_name}.jpg"), cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR))
                
                with open(os.path.join(OUTPUT_LABELS_DIR, f"{new_base_name}.txt"), 'w') as f:
                    for bbox, cls in zip(t_bboxes, t_classes):
                        f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                        
            except ValueError as e:
                print(f"Augmentation {i} per {img_name} scartata: bounding box fuori campo.")


def run_inference(
    model_path: str,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show: bool = False,
    device: str = None,
):
    
    '''
    # --- Carica il modello ---
    print(f"[INFO] Caricamento modello: {model_path}")
    model = YOLO(model_path)
    if device:
        model.to(device)

    # --- Apri il video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps — {total} frame totali")

    # --- Writer per il video di output ---
    writer = None
    if output_path:
        # Nota: output deve essere un .mp4, non un .txt
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Salvataggio risultati in: {output_path}")

    frame_idx = 0
    print("[INFO] Avvio inferenza — premi 'q' per uscire, 's' per salvare uno screenshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # --- Inferenza ---
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

        # --- Disegna le bounding box sul frame ---
        annotated = results[0].plot()

        # --- Statistiche a console ogni 30 frame ---
        if frame_idx % 30 == 0 or frame_idx == 1:
            det = results[0].boxes
            n   = len(det) if det is not None else 0
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  Frame {frame_idx:>5}/{total} ({pct:5.1f}%)  —  {n} oggetti rilevati")

        # --- Overlay info ---
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}  |  Conf >= {conf_threshold}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # --- Mostra ---
        if show:
            cv2.imshow("YOLO Inference", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Uscita richiesta dall'utente.")
                break
            if key == ord("s"):
                screenshot = f"screenshot_frame_{frame_idx}.jpg"
                cv2.imwrite(screenshot, annotated)
                print(f"[INFO] Screenshot salvato: {screenshot}")

        if writer:
            writer.write(annotated)

    # --- Pulizia ---
    cap.release()
    if writer:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    print(f"\n[DONE] Elaborati {frame_idx} frame.")
    if output_path:
        print(f"[DONE] Video salvato in: {output_path}")
'''

run_inference(
    model_path  = r"C:\Users\divic\Desktop\FVAB\esperimento_4\weights\best.pt",
    video_path  = r"C:\Users\divic\Desktop\FVAB\video_test.mp4",
    output_path = r"C:\Users\divic\Desktop\FVAB\output_esperimento_4.mp4",
    conf_threshold = 0.25,
    iou_threshold  = 0.45,
    show           = False,
    device         = None,
)


# comando per far partire report Dataset
#if __name__ == "__main__":
    # main()


# Comando per testare BB:
# visualizza_yolo_bboxes(r"C:\Users\divic\Desktop\utiliGTSDB\labels\00001.txt")

#comando per Augumentazione
#process_images(
    #img_in    = r"C:\Users\divic\Desktop\FVAB\DatasetCompleto\images",
    #img_out   = r"C:\Users\divic\Desktop\FVAB\DatasetAgumented\images",
    #lbl_in    = r"C:\Users\divic\Desktop\FVAB\DatasetCompleto\labels",
    #lbl_out   = r"C:\Users\divic\Desktop\FVAB\DatasetAgumented\labels",
    #n_mosaics        = 500,
    #n_translation    = 500,
#)

#Comando per il remapping 
#remap_labels()

#Comando per augumentationGTSDB
#process_dataset()