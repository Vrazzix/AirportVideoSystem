"""
📂 Скрипт разделения датасета на train/val

Разделяет изображения и метки на тренировочную и валидационную выборки,
сохраняя пары image + label вместе.

Использование:
    python split_dataset.py --input_images путь/к/images --input_labels путь/к/labels --output_dir путь/к/результату --val_ratio 0.2

Результат:
    output_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
"""

import argparse
import os
import shutil
import random
from pathlib import Path


def split_dataset(input_images, input_labels, output_dir, val_ratio=0.2, seed=42):
    """
    Разделяет датасет на train и val.

    Args:
        input_images: папка с изображениями
        input_labels: папка с YOLO-метками (.txt)
        output_dir:   корневая папка для результата
        val_ratio:    доля валидационных данных (0.0 - 1.0)
        seed:         seed для воспроизводимости
    """
    random.seed(seed)

    # Поддерживаемые форматы
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    # Собираем все изображения
    all_images = sorted([
        f for f in os.listdir(input_images)
        if Path(f).suffix.lower() in img_extensions
    ])

    if not all_images:
        print(f"❌ Не найдено изображений в {input_images}")
        return

    # Проверяем наличие меток
    paired = []
    missing_labels = []

    for img_file in all_images:
        stem = Path(img_file).stem
        label_file = stem + ".txt"
        label_path = os.path.join(input_labels, label_file)

        if os.path.exists(label_path):
            paired.append((img_file, label_file))
        else:
            missing_labels.append(img_file)

    print(f"📊 Статистика датасета:")
    print(f"   Всего изображений: {len(all_images)}")
    print(f"   С метками:         {len(paired)}")
    print(f"   Без меток:         {len(missing_labels)}")

    if missing_labels:
        print(f"\n⚠️  Изображения без меток ({len(missing_labels)} шт.):")
        for f in missing_labels[:10]:
            print(f"      {f}")
        if len(missing_labels) > 10:
            print(f"      ... и ещё {len(missing_labels) - 10}")
        print(f"   Эти изображения будут пропущены.\n")

    if not paired:
        print("❌ Нет ни одной пары image + label. Проверьте пути.")
        return

    # Перемешиваем и разделяем
    random.shuffle(paired)
    val_count = max(1, int(len(paired) * val_ratio))
    train_count = len(paired) - val_count

    val_pairs = paired[:val_count]
    train_pairs = paired[val_count:]

    print(f"📦 Разделение (seed={seed}):")
    print(f"   Train: {train_count} изображений ({100 - val_ratio * 100:.0f}%)")
    print(f"   Val:   {val_count} изображений ({val_ratio * 100:.0f}%)")

    # Создаём папки
    dirs = {
        "train_images": os.path.join(output_dir, "train", "images"),
        "train_labels": os.path.join(output_dir, "train", "labels"),
        "val_images":   os.path.join(output_dir, "val", "images"),
        "val_labels":   os.path.join(output_dir, "val", "labels"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Копируем файлы
    def copy_pairs(pairs, img_dst, lbl_dst, name):
        for i, (img_file, lbl_file) in enumerate(pairs):
            src_img = os.path.join(input_images, img_file)
            src_lbl = os.path.join(input_labels, lbl_file)

            shutil.copy2(src_img, os.path.join(img_dst, img_file))
            shutil.copy2(src_lbl, os.path.join(lbl_dst, lbl_file))

            if (i + 1) % 100 == 0:
                print(f"   {name}: скопировано {i + 1}/{len(pairs)}...")

    print(f"\n📁 Копирование в {output_dir}...")
    copy_pairs(train_pairs, dirs["train_images"], dirs["train_labels"], "Train")
    copy_pairs(val_pairs, dirs["val_images"], dirs["val_labels"], "Val")

    # Анализ классов в train и val
    print(f"\n📊 Распределение классов:")
    for split_name, pairs in [("Train", train_pairs), ("Val", val_pairs)]:
        class_counts = {}
        for _, lbl_file in pairs:
            lbl_path = os.path.join(input_labels, lbl_file)
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1

        print(f"   {split_name}:")
        if class_counts:
            for cls_id in sorted(class_counts.keys()):
                print(f"      Класс {cls_id}: {class_counts[cls_id]} объектов")
        else:
            print(f"      (пусто)")

    # Генерируем data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_path = os.path.abspath(output_dir).replace("\\", "/")

    yaml_content = f"""# Auto-generated data.yaml
path: {abs_path}
train: train/images
val: val/images

nc: 2
names: ['chock', 'wheel']

# ─────────────────────────────────────────
# После аугментации train замените путь:
# train: train_augmented/images
# ─────────────────────────────────────────
#
# Команда обучения:
# yolo detect train data={yaml_path} model=yolov8s.pt epochs=100 imgsz=1280 batch=8
"""

    # Замените на:
with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n✅ Готово!")
    print(f"   📁 Train: {dirs['train_images']}")
    print(f"   📁 Val:   {dirs['val_images']}")
    print(f"   📝 Config: {yaml_path}")
    print(f"\n🔜 Следующий шаг — аугментация train:")
    print(f'   python augmentation.py --input_dir "{dirs["train_images"]}" --label_dir "{dirs["train_labels"]}" --output_dir "{os.path.join(output_dir, "train_augmented")}" --num_aug 10 --gen_config')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val")
    parser.add_argument("--input_images", required=True, help="Папка с изображениями")
    parser.add_argument("--input_labels", required=True, help="Папка с YOLO-метками (.txt)")
    parser.add_argument("--output_dir", required=True, help="Корневая папка для результата")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Доля val (по умолчанию 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")

    args = parser.parse_args()

    split_dataset(
        input_images=args.input_images,
        input_labels=args.input_labels,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
