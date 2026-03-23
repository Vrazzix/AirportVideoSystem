"""
🔧 Скрипт аугментации данных для дообучения модели колёс и колодок

Решает проблемы:
1. Колодки не видны при перекрытии человеком → аугментация с наложением силуэтов
2. Колёса путаются с круглыми объектами (двигатели, буквы) → hard negatives
3. Крупный/мелкий план не распознаётся → мультимасштабная аугментация
4. Колодки в руках не детектируются → копирование колодок в новые контексты

Использование:
    python augmentation.py --input_dir data/images --label_dir data/labels --output_dir data/augmented

Требования:
    pip install albumentations opencv-python-headless numpy Pillow
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil

try:
    import albumentations as A
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("⚠️ albumentations не установлен. pip install albumentations")


# ══════════════════════════════════════════════
# 1. БАЗОВЫЕ АУГМЕНТАЦИИ (albumentations)
# ══════════════════════════════════════════════
def get_base_augmentations():
    """Базовые аугментации для улучшения робастности модели."""
    return A.Compose([
        # Геометрические
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.3,        # ★ Широкий диапазон масштаба!
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.Perspective(scale=(0.05, 0.12), p=0.3),

        # Яркость / контраст / цвет
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.CLAHE(clip_limit=4.0, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1),
        ], p=0.8),

        # Погода / условия
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200,200,200), blur_value=3, brightness_coefficient=0.8, p=1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 3), shadow_dimension=5, p=1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=3, num_flare_circles_upper=6, src_radius=100, p=1),
        ], p=0.3),

        # Размытие / шум
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.GaussNoise(var_limit=(10, 50), p=1),
        ], p=0.3),

        # Качество (имитация сжатия камеры)
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


# ══════════════════════════════════════════════
# 2. МУЛЬТИМАСШТАБНАЯ АУГМЕНТАЦИЯ
# ══════════════════════════════════════════════
def get_multiscale_augmentations():
    """
    ★ Решает проблему: модель не видит колёса/колодки при крупном и мелком плане.
    Создаёт вариации с сильным зумом и уменьшением.
    """
    augmentations = []

    # Сильный zoom-in (крупный план) — имитация камеры вблизи
    augmentations.append(("zoom_in", A.Compose([
        A.RandomCrop(height=480, width=640, p=0.5),
        A.Resize(height=720, width=1280),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))))

    # Zoom-out 2x (мелкий план) — имитация дальней камеры
    augmentations.append(("zoom_out", A.Compose([
        A.PadIfNeeded(min_height=1440, min_width=2560, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(height=720, width=1280),
        A.RandomBrightnessContrast(brightness_limit=0.15, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))))

    # Zoom-out 4x (очень дальний план) — колодки становятся совсем маленькими
    augmentations.append(("zoom_out_far", A.Compose([
        A.PadIfNeeded(min_height=2880, min_width=5120, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(height=720, width=1280),
        A.RandomBrightnessContrast(brightness_limit=0.2, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Лёгкое размытие как на дальнем плане
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.05))))

    # Zoom-out 3x с шумом — имитация сжатой/удалённой камеры
    augmentations.append(("zoom_out_noisy", A.Compose([
        A.PadIfNeeded(min_height=2160, min_width=3840, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(height=720, width=1280),
        A.GaussNoise(var_limit=(20, 60), p=0.7),
        A.ImageCompression(quality_lower=40, quality_upper=70, p=0.5),  # Артефакты сжатия
        A.RandomBrightnessContrast(brightness_limit=0.15, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.05))))

    return augmentations


# ══════════════════════════════════════════════
# 3. COPY-PASTE АУГМЕНТАЦИЯ (колодки в новых контекстах)
# ══════════════════════════════════════════════
def copy_paste_chock(image, bboxes, class_labels, chock_cls_id=0):
    """
    ★ Решает: колодки в руках / колодки в нетипичных местах.
    Копирует вырезанные колодки в случайные места кадра.
    """
    h, w = image.shape[:2]
    result = image.copy()
    new_bboxes = list(bboxes)
    new_labels = list(class_labels)

    # Найти колодки в текущих bbox
    chock_crops = []
    for bbox, cls in zip(bboxes, class_labels):
        if cls == chock_cls_id:
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                chock_crops.append(result[y1:y2, x1:x2].copy())

    if not chock_crops:
        return result, new_bboxes, new_labels

    # Вставить 1-3 копии в случайные места
    for _ in range(random.randint(1, 3)):
        crop = random.choice(chock_crops)
        ch, cw = crop.shape[:2]

        # Случайный масштаб
        scale = random.uniform(0.5, 1.5)
        new_cw, new_ch = int(cw * scale), int(ch * scale)
        if new_cw < 10 or new_ch < 10:
            continue
        crop_resized = cv2.resize(crop, (new_cw, new_ch))

        # Случайное размещение (предпочтительно в нижней части — у колёс)
        paste_x = random.randint(0, max(0, w - new_cw))
        paste_y = random.randint(int(h * 0.4), max(int(h * 0.4) + 1, h - new_ch))

        if paste_y + new_ch > h or paste_x + new_cw > w:
            continue

        # Смешивание (лёгкое)
        alpha = random.uniform(0.7, 1.0)
        roi = result[paste_y:paste_y + new_ch, paste_x:paste_x + new_cw]
        blended = cv2.addWeighted(crop_resized, alpha, roi, 1 - alpha, 0)
        result[paste_y:paste_y + new_ch, paste_x:paste_x + new_cw] = blended

        # Добавляем bbox (YOLO format)
        new_cx = (paste_x + new_cw / 2) / w
        new_cy = (paste_y + new_ch / 2) / h
        new_bw = new_cw / w
        new_bh = new_ch / h
        new_bboxes.append([new_cx, new_cy, new_bw, new_bh])
        new_labels.append(chock_cls_id)

    return result, new_bboxes, new_labels


# ══════════════════════════════════════════════
# 4. HARD NEGATIVE MINING HELPER
# ══════════════════════════════════════════════
def create_hard_negative_list():
    """
    ★ Решает: двигатели/буквы/круглые объекты распознаются как колёса.

    Инструкция по созданию hard negatives:
    1. Соберите изображения БЕЗ колёс/колодок, но С круглыми объектами:
       - Двигатели самолётов крупным планом
       - Текст с буквами O, Q, 0
       - Дорожные конусы
       - Круглые знаки, фары, иллюминаторы
    2. Создайте пустые label-файлы (без bbox) для этих изображений
    3. Добавьте в тренировочный датасет

    Это научит модель НЕ реагировать на похожие объекты.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║           📋 ИНСТРУКЦИЯ: Hard Negative Mining               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Чтобы модель перестала путать двигатели/буквы с колёсами:   ║
║                                                              ║
║  1. Создайте папку hard_negatives/images/                    ║
║  2. Соберите туда 50-200 изображений с:                      ║
║     • Двигатели самолётов (вид сбоку, спереди)               ║
║     • Текст крупным планом (особенно буквы O, Q)             ║
║     • Дорожные конусы, фары, круглые знаки                   ║
║     • Иллюминаторы самолёта                                  ║
║  3. Для каждого — ПУСТОЙ .txt файл в labels/                 ║
║  4. Добавьте в data.yaml как часть train                     ║
║                                                              ║
║  Это «учит» модель, что эти объекты — НЕ колёса.            ║
╚══════════════════════════════════════════════════════════════╝
    """)


# ══════════════════════════════════════════════
# 5. OCCLUSION AUGMENTATION
# ══════════════════════════════════════════════
def add_person_occlusion(image, bboxes, class_labels, chock_cls_id=0):
    """
    ★ Решает: колодки не видны при перекрытии человеком.
    Накладывает случайные прямоугольники (имитация ног/тела) поверх колодок.
    Это учит модель видеть частично перекрытые объекты.
    """
    h, w = image.shape[:2]
    result = image.copy()

    for bbox, cls in zip(bboxes, class_labels):
        if cls == chock_cls_id and random.random() < 0.4:  # 40% шанс перекрытия
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Случайный прямоугольник-перекрытие (имитация ноги)
            occ_w = int((x2 - x1) * random.uniform(0.3, 0.8))
            occ_h = int((y2 - y1) * random.uniform(1.5, 3.0))
            occ_x = random.randint(x1, max(x1 + 1, x2 - occ_w))
            occ_y = max(0, y1 - occ_h // 2)

            # Цвет (тёмный, как одежда)
            color = (
                random.randint(20, 80),
                random.randint(20, 80),
                random.randint(20, 80),
            )
            cv2.rectangle(result, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), color, -1)

    return result


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════
def read_yolo_labels(label_path):
    """Read YOLO format labels."""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                bboxes.append([cx, cy, bw, bh])
                class_labels.append(cls)
    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    """Write YOLO format labels."""
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            cx, cy, bw, bh = bbox
            # Clamp values
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def run_augmentation(input_dir, label_dir, output_dir, num_augmented=5, chock_cls=0):
    """
    Main augmentation pipeline.

    Args:
        input_dir:  папка с исходными изображениями
        label_dir:  папка с YOLO-метками (.txt)
        output_dir: папка для аугментированных данных
        num_augmented: сколько вариантов каждого изображения создать
        chock_cls: ID класса колодки
    """
    out_img_dir = os.path.join(output_dir, "images")
    out_lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not HAS_ALBUM:
        print("❌ Установите albumentations: pip install albumentations")
        return

    base_aug = get_base_augmentations()
    multiscale_augs = get_multiscale_augmentations()

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    print(f"📂 Найдено изображений: {len(image_files)}")
    print(f"📝 Аугментаций на изображение: {num_augmented}")
    print(f"📊 Итого будет: ~{len(image_files) * num_augmented} новых изображений")
    print()

    total_created = 0

    for img_idx, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)
        stem = Path(img_file).stem
        lbl_path = os.path.join(label_dir, stem + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ⚠️ Не удалось прочитать: {img_file}")
            continue

        bboxes, class_labels = read_yolo_labels(lbl_path)

        # Копируем оригинал
        shutil.copy(img_path, os.path.join(out_img_dir, img_file))
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(out_lbl_dir, stem + ".txt"))

        for aug_idx in range(num_augmented):
            try:
                aug_image = image.copy()
                aug_bboxes = [list(b) for b in bboxes]
                aug_labels = list(class_labels)

                # Step 1: Copy-paste колодок (30% шанс)
                if random.random() < 0.3 and aug_bboxes:
                    aug_image, aug_bboxes, aug_labels = copy_paste_chock(
                        aug_image, aug_bboxes, aug_labels, chock_cls
                    )

                # Step 2: Перекрытие человеком (25% шанс)
                if random.random() < 0.25:
                    aug_image = add_person_occlusion(aug_image, aug_bboxes, aug_labels, chock_cls)

                # Step 3: Базовые аугментации
                if aug_bboxes:
                    result = base_aug(
                        image=aug_image,
                        bboxes=aug_bboxes,
                        class_labels=aug_labels,
                    )
                    aug_image = result['image']
                    aug_bboxes = result['bboxes']
                    aug_labels = result['class_labels']

                # Step 4: Мультимасштаб (каждая 2-я итерация — чаще для дальнего плана)
                if aug_idx % 2 == 0 and aug_bboxes:
                    ms_name, ms_aug = random.choice(multiscale_augs)
                    try:
                        ms_result = ms_aug(
                            image=aug_image,
                            bboxes=aug_bboxes,
                            class_labels=aug_labels,
                        )
                        aug_image = ms_result['image']
                        aug_bboxes = ms_result['bboxes']
                        aug_labels = ms_result['class_labels']
                    except Exception:
                        pass  # Некоторые масштабы могут не сработать

                # Сохранение
                out_name = f"{stem}_aug{aug_idx}"
                out_img_path = os.path.join(out_img_dir, out_name + ".jpg")
                out_lbl_path = os.path.join(out_lbl_dir, out_name + ".txt")

                cv2.imwrite(out_img_path, aug_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                write_yolo_labels(out_lbl_path, aug_bboxes, aug_labels)
                total_created += 1

            except Exception as e:
                print(f"  ⚠️ Ошибка aug #{aug_idx} для {img_file}: {e}")

        if (img_idx + 1) % 10 == 0:
            print(f"  ✅ {img_idx + 1}/{len(image_files)} обработано...")

    print(f"\n🎉 Готово! Создано {total_created} аугментированных изображений")
    print(f"📁 Результат: {output_dir}")

    # Show hard negative instructions
    create_hard_negative_list()


# ══════════════════════════════════════════════
# TRAINING CONFIG GENERATOR
# ══════════════════════════════════════════════
def generate_training_config(output_dir, num_classes=2, class_names=None):
    """Generate data.yaml for YOLOv8 training."""
    if class_names is None:
        class_names = ["chock", "wheel"]

    yaml_content = f"""# YOLOv8 Training Config — Auto-generated
# ─────────────────────────────────────────

path: {os.path.abspath(output_dir)}
train: images
val: images  # В идеале разделите на train/val (80/20)

nc: {num_classes}
names: {class_names}

# Рекомендуемые параметры обучения:
# yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=1280 batch=8
# 
# ★ КЛЮЧЕВЫЕ ПАРАМЕТРЫ для вашего случая:
# - imgsz=1280 (большие изображения для мелких объектов)
# - mosaic=1.0 (включить мозаику)  
# - mixup=0.15 (смешивание изображений)
# - copy_paste=0.3 (встроенная copy-paste аугментация)
# - degrees=15 (вращение)
# - scale=0.5 (масштаб ±50%)
# - perspective=0.001 (перспективное искажение)
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"📝 Конфиг сохранён: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation for aircraft chock/wheel detection")
    parser.add_argument("--input_dir", required=True, help="Папка с исходными изображениями")
    parser.add_argument("--label_dir", required=True, help="Папка с YOLO-метками (.txt)")
    parser.add_argument("--output_dir", required=True, help="Папка для результата")
    parser.add_argument("--num_aug", type=int, default=5, help="Кол-во аугментаций на изображение")
    parser.add_argument("--chock_cls", type=int, default=0, help="ID класса колодки")
    parser.add_argument("--gen_config", action="store_true", help="Сгенерировать data.yaml")

    args = parser.parse_args()

    run_augmentation(
        input_dir=args.input_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        num_augmented=args.num_aug,
        chock_cls=args.chock_cls,
    )

    if args.gen_config:
        generate_training_config(args.output_dir)
