"""
🎓 Auto-Labeling: Модель-учитель размечает новые данные

Ваша обученная модель (учитель) генерирует YOLO-разметку для новых изображений.
Затем вы проверяете/исправляете разметку и обучаете модель-ученика.

Использование:
    python auto_label.py \
        --model "путь/к/bestBoots_v2.pt" \
        --input_dir "downloaded_images/chocks" \
        --output_dir "auto_labeled" \
        --conf 0.3 \
        --imgsz 1280

Пайплайн:
    1. auto_label.py   → генерация псевдо-меток
    2. Проверка в CVAT/LabelImg → исправление ошибок
    3. split_dataset.py → разделение train/val
    4. augmentation.py  → аугментация train
    5. Обучение ученика
"""

import argparse
import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("❌ ultralytics не установлен: pip install ultralytics")

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ══════════════════════════════════════════════
# 1. ОСНОВНОЙ AUTO-LABELER
# ══════════════════════════════════════════════

def auto_label_images(
    model_path,
    input_dir,
    output_dir,
    conf_threshold=0.3,
    imgsz=1280,
    high_conf=0.7,
    low_conf=0.3,
    classes_map=None,
):
    """
    Запускает модель-учителя на всех изображениях и создаёт YOLO-метки.

    Изображения сортируются по 3 категориям:
    - high_confidence/  — conf ≥ high_conf (можно использовать без проверки)
    - review_needed/    — low_conf ≤ conf < high_conf (нужна проверка)
    - no_detections/    — ничего не найдено (разметить вручную или выбросить)

    Args:
        model_path:     путь к .pt модели-учителя
        input_dir:      папка с неразмеченными изображениями
        output_dir:     папка для результата
        conf_threshold: минимальный порог (ниже — отбрасываем)
        imgsz:          размер входа модели
        high_conf:      порог «высокая уверенность» (авто-принятие)
        low_conf:       порог «низкая уверенность» (нужна проверка)
        classes_map:    dict {old_id: new_id} для ремаппинга классов
    """
    if not HAS_YOLO:
        print("❌ Установите ultralytics: pip install ultralytics")
        return

    # Загрузка модели
    print(f"📦 Загрузка модели: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"   Классы: {class_names}")

    # Создание директорий
    dirs = {}
    for category in ["high_confidence", "review_needed", "no_detections"]:
        img_dir = os.path.join(output_dir, category, "images")
        lbl_dir = os.path.join(output_dir, category, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        dirs[category] = {"images": img_dir, "labels": lbl_dir}

    # Папка для визуализаций (проверка глазами)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Собираем изображения (рекурсивно по всем подпапкам)
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_files = []
    for root, subdirs, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in img_extensions:
                # Сохраняем полный путь относительно input_dir
                full_path = os.path.join(root, f)
                image_files.append(full_path)
    image_files.sort()

    if not image_files:
        print(f"❌ Не найдено изображений в {input_dir} (включая подпапки)")
        print(f"   Проверьте путь и наличие файлов .jpg/.png/.bmp")
        # Показываем содержимое папки для диагностики
        if os.path.exists(input_dir):
            contents = os.listdir(input_dir)
            print(f"   Содержимое папки: {contents[:15]}")
        return

    print(f"\n🔍 Найдено изображений: {len(image_files)}")
    print(f"   Пороги: high ≥ {high_conf}, review ≥ {low_conf}, drop < {low_conf}")
    print(f"   imgsz: {imgsz}")

    # Статистика
    stats = {
        "total": len(image_files),
        "high_confidence": 0,
        "review_needed": 0,
        "no_detections": 0,
        "total_objects": 0,
        "class_counts": {},
        "conf_distribution": [],
    }

    # Обработка
    for idx, img_path in enumerate(image_files):
        img_file = os.path.basename(img_path)  # только имя файла
        stem = Path(img_file).stem

        # Инференс
        results = model(img_path, conf=conf_threshold, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # Нет детекций
            category = "no_detections"
            stats["no_detections"] += 1
            shutil.copy2(img_path, os.path.join(dirs[category]["images"], img_file))
            # Пустой label-файл
            open(os.path.join(dirs[category]["labels"], stem + ".txt"), 'w').close()

        else:
            # Есть детекции — определяем категорию
            confs = boxes.conf.cpu().numpy()
            min_conf = float(confs.min())
            avg_conf = float(confs.mean())

            # Категория определяется по средней уверенности
            if avg_conf >= high_conf:
                category = "high_confidence"
                stats["high_confidence"] += 1
            else:
                category = "review_needed"
                stats["review_needed"] += 1

            # Сохраняем изображение
            shutil.copy2(img_path, os.path.join(dirs[category]["images"], img_file))

            # Генерируем YOLO-метки
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            label_lines = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())

                # Ремаппинг классов (если нужно)
                if classes_map and cls_id in classes_map:
                    cls_id = classes_map[cls_id]

                # YOLO format: class cx cy bw bh
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                # Clamp
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                bw = max(0, min(1, bw))
                bh = max(0, min(1, bh))

                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # Статистика
                stats["total_objects"] += 1
                cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                stats["class_counts"][cls_name] = stats["class_counts"].get(cls_name, 0) + 1
                stats["conf_distribution"].append(conf_val)

            # Сохраняем метки
            label_path = os.path.join(dirs[category]["labels"], stem + ".txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))

            # Визуализация (каждое 5-е + все review_needed)
            if category == "review_needed" or idx % 5 == 0:
                viz_img = img.copy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf_val = float(box.conf[0].cpu().numpy())
                    cls_name = class_names.get(cls_id, f"cls_{cls_id}")

                    color = (0, 255, 0) if conf_val >= high_conf else (0, 165, 255)
                    cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                    text = f"{cls_name} {conf_val:.2f}"
                    cv2.putText(viz_img, text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Плашка с категорией
                badge_color = (0, 180, 0) if category == "high_confidence" else (0, 140, 255)
                cv2.rectangle(viz_img, (0, 0), (350, 35), badge_color, -1)
                cv2.putText(viz_img, f"{category} | avg:{avg_conf:.2f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                viz_path = os.path.join(viz_dir, f"{category}_{img_file}")
                cv2.imwrite(viz_path, viz_img)

        # Прогресс
        if (idx + 1) % 50 == 0 or idx == len(image_files) - 1:
            print(f"   ✅ {idx + 1}/{len(image_files)} обработано...")

    # ── Отчёт ──
    print(f"\n{'='*60}")
    print(f"📊 ОТЧЁТ AUTO-LABELING")
    print(f"{'='*60}")
    print(f"   Всего изображений:     {stats['total']}")
    print(f"   ✅ Высокая уверенность: {stats['high_confidence']} (conf ≥ {high_conf})")
    print(f"   ⚠️  Нужна проверка:     {stats['review_needed']} ({low_conf} ≤ conf < {high_conf})")
    print(f"   ❌ Без детекций:        {stats['no_detections']}")
    print(f"   📦 Всего объектов:      {stats['total_objects']}")

    if stats["class_counts"]:
        print(f"\n   Распределение классов:")
        for cls_name, count in sorted(stats["class_counts"].items()):
            print(f"      {cls_name}: {count}")

    if stats["conf_distribution"]:
        confs = stats["conf_distribution"]
        print(f"\n   Уверенность модели:")
        print(f"      Мин:     {min(confs):.3f}")
        print(f"      Средняя: {np.mean(confs):.3f}")
        print(f"      Макс:    {max(confs):.3f}")
        print(f"      Медиана: {np.median(confs):.3f}")

    # Сохраняем статистику в JSON
    stats["conf_distribution"] = []  # Не сохраняем весь массив
    stats_path = os.path.join(output_dir, "labeling_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Результат:")
    print(f"   {output_dir}/")
    print(f"   ├── high_confidence/    ← можно сразу в обучение")
    print(f"   │   ├── images/")
    print(f"   │   └── labels/")
    print(f"   ├── review_needed/      ← проверить в LabelImg/CVAT")
    print(f"   │   ├── images/")
    print(f"   │   └── labels/")
    print(f"   ├── no_detections/      ← разметить вручную или выбросить")
    print(f"   │   ├── images/")
    print(f"   │   └── labels/")
    print(f"   ├── visualizations/     ← визуальная проверка")
    print(f"   └── labeling_stats.json")

    print(f"\n🔜 Следующие шаги:")
    print(f"   1. Просмотрите visualizations/ — убедитесь что разметка корректна")
    print(f"   2. Откройте review_needed/ в LabelImg и исправьте ошибки")
    print(f"   3. Решите что делать с no_detections/ (разметить или удалить)")
    print(f"   4. Объедините high_confidence + исправленные review_needed")
    print(f"   5. Добавьте к существующему train датасету")
    print(f"   6. split_dataset.py → augmentation.py → обучение ученика")


# ══════════════════════════════════════════════
# 2. ОБЪЕДИНЕНИЕ ПРОВЕРЕННЫХ ДАННЫХ
# ══════════════════════════════════════════════

def merge_labeled_data(auto_labeled_dir, existing_train_dir, merged_dir):
    """
    Объединяет автоматически размеченные данные с существующим train.

    Args:
        auto_labeled_dir: папка high_confidence/ (или уже проверенные)
        existing_train_dir: существующий train/ с images/ и labels/
        merged_dir: папка для объединённого результата
    """
    merged_img = os.path.join(merged_dir, "images")
    merged_lbl = os.path.join(merged_dir, "labels")
    os.makedirs(merged_img, exist_ok=True)
    os.makedirs(merged_lbl, exist_ok=True)

    count = 0

    # Копируем существующие данные
    src_imgs = os.path.join(existing_train_dir, "images")
    src_lbls = os.path.join(existing_train_dir, "labels")

    if os.path.exists(src_imgs):
        for f in os.listdir(src_imgs):
            shutil.copy2(os.path.join(src_imgs, f), os.path.join(merged_img, f))
            count += 1

    if os.path.exists(src_lbls):
        for f in os.listdir(src_lbls):
            shutil.copy2(os.path.join(src_lbls, f), os.path.join(merged_lbl, f))

    print(f"   Скопировано из train: {count} изображений")

    # Добавляем auto-labeled
    auto_imgs = os.path.join(auto_labeled_dir, "images")
    auto_lbls = os.path.join(auto_labeled_dir, "labels")
    added = 0

    if os.path.exists(auto_imgs):
        for f in os.listdir(auto_imgs):
            dst = os.path.join(merged_img, f)
            if not os.path.exists(dst):  # Не перезаписываем
                shutil.copy2(os.path.join(auto_imgs, f), dst)
                added += 1

                # Метка
                stem = Path(f).stem
                lbl_src = os.path.join(auto_lbls, stem + ".txt")
                if os.path.exists(lbl_src):
                    shutil.copy2(lbl_src, os.path.join(merged_lbl, stem + ".txt"))

    print(f"   Добавлено auto-labeled: {added} изображений")
    print(f"   Итого: {count + added} изображений в {merged_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label images using teacher model")
    parser.add_argument("--model", required=True, help="Путь к .pt модели-учителя")
    parser.add_argument("--input_dir", required=True, help="Папка с неразмеченными изображениями")
    parser.add_argument("--output_dir", default="auto_labeled", help="Папка для результата")
    parser.add_argument("--conf", type=float, default=0.3, help="Мин. порог уверенности")
    parser.add_argument("--imgsz", type=int, default=1280, help="Размер входа модели")
    parser.add_argument("--high_conf", type=float, default=0.7, help="Порог высокой уверенности")
    parser.add_argument("--low_conf", type=float, default=0.3, help="Порог низкой уверенности")

    # Merge mode
    parser.add_argument("--merge", action="store_true", help="Режим объединения данных")
    parser.add_argument("--auto_dir", help="Папка с проверенными auto-labeled данными")
    parser.add_argument("--train_dir", help="Существующий train/")
    parser.add_argument("--merged_dir", help="Папка для объединённого результата")

    args = parser.parse_args()

    if args.merge:
        if not all([args.auto_dir, args.train_dir, args.merged_dir]):
            print("❌ Для --merge нужны: --auto_dir, --train_dir, --merged_dir")
        else:
            merge_labeled_data(args.auto_dir, args.train_dir, args.merged_dir)
    else:
        auto_label_images(
            model_path=args.model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            high_conf=args.high_conf,
            low_conf=args.low_conf,
        )
