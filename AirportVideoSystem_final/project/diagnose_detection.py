"""
Диагностика: почему модель не видит колёса/колодки.
Запуск:
    python diagnose_detection.py --image path/to/frame.jpg
    python diagnose_detection.py --video path/to/video.mp4 --frame 120
"""
import argparse
import os
import sys
import cv2
import numpy as np

MODEL_PATH = r"C:\Users\shche\Desktop\Application_for_models\models\BestBoots_v2.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Tiled inference — делим кадр на тайлы, запускаем модель на каждом
# ─────────────────────────────────────────────────────────────────────────────

def tile_image(img, tile_size=640, overlap=0.25):
    """Нарезает img на тайлы tile_size×tile_size с перекрытием overlap."""
    H, W = img.shape[:2]
    step = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((img[y1:y2, x1:x2], x1, y1))
    return tiles


def nms_boxes(boxes, scores, iou_thr=0.5):
    """Simple NMS over merged tile results."""
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def tiled_predict(model, img, conf=0.15, tile_size=640, overlap=0.25, imgsz=640):
    """
    Запускает модель на тайлах и возвращает боксы в координатах исходного img.
    """
    tiles = tile_image(img, tile_size=tile_size, overlap=overlap)
    all_boxes, all_scores, all_classes = [], [], []

    for tile, ox, oy in tiles:
        results = model(tile, imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box, score, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                    r.boxes.conf.cpu().numpy(),
                                    r.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box
            all_boxes.append([x1+ox, y1+oy, x2+ox, y2+oy])
            all_scores.append(float(score))
            all_classes.append(cls)

    if not all_boxes:
        return [], [], []

    keep = nms_boxes(all_boxes, all_scores, iou_thr=0.5)
    return ([all_boxes[i]  for i in keep],
            [all_scores[i] for i in keep],
            [all_classes[i] for i in keep])


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_circular_window(img):
    """
    Если кадр снят через круглый иллюминатор — вырезаем центральную область,
    убирая тёмную рамку. Ищем наибольший вписанный прямоугольник.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    # Найти bounding rect ненулевой области
    coords  = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    # Небольшой отступ внутрь
    pad = int(min(w, h) * 0.03)
    return img[y+pad:y+h-pad, x+pad:x+w-pad]


def enhance_contrast(img):
    """CLAHE для улучшения локального контраста."""
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(img, model, label="frame"):
    H, W = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"  Diagnostic: {label}  ({W}x{H})")
    print(f"{'='*60}")

    results_table = []

    # ── 1. Обычный inference при разных conf ─────────────────────────────────
    print("\n[1] Standard inference at different conf thresholds:")
    for conf in [0.50, 0.25, 0.10, 0.05, 0.01]:
        res = model(img, imgsz=640, conf=conf, verbose=False)[0]
        n   = len(res.boxes) if res.boxes is not None else 0
        print(f"    conf={conf:.2f}  →  {n} detections")
        results_table.append(('standard', conf, n))

    # ── 2. С большим imgsz ───────────────────────────────────────────────────
    print("\n[2] Standard inference with larger imgsz (1280):")
    for conf in [0.25, 0.10, 0.05]:
        res = model(img, imgsz=1280, conf=conf, verbose=False)[0]
        n   = len(res.boxes) if res.boxes is not None else 0
        print(f"    conf={conf:.2f}  imgsz=1280  →  {n} detections")

    # ── 3. Обрезка иллюминатора ──────────────────────────────────────────────
    cropped = crop_circular_window(img)
    Hc, Wc  = cropped.shape[:2]
    print(f"\n[3] After circular crop ({Wc}x{Hc}):")
    for conf in [0.25, 0.10, 0.05]:
        res = model(cropped, imgsz=640, conf=conf, verbose=False)[0]
        n   = len(res.boxes) if res.boxes is not None else 0
        print(f"    conf={conf:.2f}  →  {n} detections")

    # ── 4. CLAHE contrast enhancement ────────────────────────────────────────
    enhanced = enhance_contrast(img)
    print("\n[4] After CLAHE contrast enhancement:")
    for conf in [0.25, 0.10, 0.05]:
        res = model(enhanced, imgsz=640, conf=conf, verbose=False)[0]
        n   = len(res.boxes) if res.boxes is not None else 0
        print(f"    conf={conf:.2f}  →  {n} detections")

    # ── 5. Tiled inference ───────────────────────────────────────────────────
    print("\n[5] Tiled inference (tile=640, overlap=25%):")
    for conf in [0.25, 0.10, 0.05]:
        boxes, scores, classes = tiled_predict(model, img, conf=conf,
                                               tile_size=640, overlap=0.25)
        print(f"    conf={conf:.2f}  →  {len(boxes)} detections after NMS")

    # ── 6. Tiled на обрезанном ───────────────────────────────────────────────
    print("\n[6] Tiled inference on CROPPED frame:")
    boxes, scores, classes = tiled_predict(model, cropped, conf=0.10,
                                           tile_size=640, overlap=0.3)
    print(f"    conf=0.10  →  {len(boxes)} detections")

    # ── Визуализация лучшего результата ─────────────────────────────────────
    best_img  = cropped.copy()
    best_conf = 0.05
    boxes, scores, classes = tiled_predict(model, cropped, conf=best_conf,
                                           tile_size=640, overlap=0.3)
    names = model.names

    for (x1,y1,x2,y2), score, cls in zip(boxes, scores, classes):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(best_img, (x1,y1), (x2,y2), (0,255,0), 2)
        lbl = f"{names.get(cls, cls)} {score:.2f}"
        cv2.putText(best_img, lbl, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out_path = f"diagnostic_{label.replace(' ','_')}.jpg"
    cv2.imwrite(out_path, best_img)
    print(f"\n  Saved visualization → {out_path}")

    # ── Рекомендации ─────────────────────────────────────────────────────────
    total_found = len(boxes)
    print(f"\n{'─'*60}")
    print("  РЕКОМЕНДАЦИИ:")
    if total_found > 0:
        print(f"  ✓ Tiled inference нашёл {total_found} объектов!")
        print("    → Используй tiled_predict() вместо обычного model()")
    else:
        print("  ✗ Ни один метод не нашёл объекты.")
        print("  Вероятные причины:")
        print("  1. Domain gap — модель не видела таких ракурсов при обучении")
        print("     Решение: дообучить на подобных кадрах (fine-tuning)")
        print("  2. Объекты слишком маленькие для текущего имgsz")
        print("     Решение: использовать imgsz=1280 + tiled inference")
        print("  3. Нужно добавить аугментацию: perspective, fisheye, dark frames")
    print(f"{'─'*60}")

    return boxes, scores, classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default=None)
    p.add_argument('--video', default=None)
    p.add_argument('--frame', type=int, default=0, help='Frame index from video')
    p.add_argument('--model', default=MODEL_PATH)
    args = p.parse_args()

    from ultralytics import YOLO
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Cannot read: {args.image}"); return
        run_diagnostic(img, model, label=os.path.basename(args.image))

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, img = cap.read()
        cap.release()
        if not ret:
            print("Cannot read frame"); return
        run_diagnostic(img, model, label=f"frame_{args.frame}")

    else:
        print("Provide --image or --video")


if __name__ == "__main__":
    main()
