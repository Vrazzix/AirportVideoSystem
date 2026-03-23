"""
Дообучение модели колодок/колёс на новых данных (дальний план)

Использование:
    python train_finetune.py
    python train_finetune.py --data augm_split/data.yaml --epochs 80 --batch 4

После обучения автоматически запускается валидация и сохраняется отчёт.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════

DEFAULT_CONFIG = {
    "model":   "C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestBoots_v2.pt",   # базовая модель
    "data":    "C:\\Users\\shche\\Desktop\\Application_for_models\\augm_split\\data.yaml",     # датасет с дальним планом
    "epochs":  50,
    "imgsz":   1280,
    "lr0":     0.001,      # малый LR — не "забываем" старые веса
    "freeze":  10,         # заморозить первые 10 слоёв backbone
    "batch":   8,
    "patience": 15,        # ранняя остановка если нет улучшений
    "workers": 4,
    "device":  "",         # "" = авто (GPU если есть, иначе CPU)
    "project": "runs/finetune_far",
    "name":    "boots_distant",
}


# ══════════════════════════════════════════════
# ПРОВЕРКА ОКРУЖЕНИЯ
# ══════════════════════════════════════════════

def check_environment(config):
    """Проверяет наличие всех нужных файлов перед стартом."""
    errors = []

    model_path = Path(config["model"])
    if not model_path.exists():
        errors.append(f"Модель не найдена: {model_path}")

    data_path = Path(config["data"])
    if not data_path.exists():
        errors.append(f"data.yaml не найден: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        errors.append("ultralytics не установлен: pip install ultralytics")

    if errors:
        print("\n❌ Ошибки конфигурации:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)

    print("✅ Окружение проверено")


# ══════════════════════════════════════════════
# ОБУЧЕНИЕ
# ══════════════════════════════════════════════

def run_training(config):
    """Запускает дообучение модели."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"🚀 ДООБУЧЕНИЕ МОДЕЛИ")
    print(f"{'='*60}")
    print(f"   Базовая модель : {config['model']}")
    print(f"   Датасет        : {config['data']}")
    print(f"   Эпох           : {config['epochs']}")
    print(f"   Размер изобр.  : {config['imgsz']}")
    print(f"   Learning rate  : {config['lr0']}")
    print(f"   Заморозка слоёв: {config['freeze']}")
    print(f"   Batch size     : {config['batch']}")
    print(f"   Ранняя остановка: {config['patience']} эпох")
    print(f"{'='*60}\n")

    model = YOLO(config["model"])

    start_time = time.time()

    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        lr0=config["lr0"],
        freeze=config["freeze"],
        batch=config["batch"],
        patience=config["patience"],
        workers=config["workers"],
        device=config["device"] if config["device"] else None,
        project=config["project"],
        name=config["name"],
        # Аугментации — усиленные для мелких объектов
        mosaic=1.0,          # мозаика (несколько изображений в одно)
        mixup=0.15,          # смешивание изображений
        copy_paste=0.3,      # встроенная copy-paste аугментация
        degrees=15,          # вращение
        scale=0.5,           # масштаб ±50%
        perspective=0.001,   # перспективное искажение
        # Оптимизатор
        optimizer="AdamW",
        weight_decay=0.0005,
        warmup_epochs=3,
        # Сохранение
        save=True,
        save_period=10,      # сохранять чекпоинт каждые 10 эпох
        exist_ok=True,
    )

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    print(f"\n✅ Обучение завершено за {elapsed_str}")

    # Путь к лучшей модели
    best_model_path = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"💾 Лучшая модель: {best_model_path}")
    else:
        # Fallback: ищем через results
        try:
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        except Exception:
            best_model_path = None

    return results, best_model_path, elapsed


# ══════════════════════════════════════════════
# ВАЛИДАЦИЯ
# ══════════════════════════════════════════════

def run_validation(best_model_path, config):
    """Запускает валидацию на лучшей модели и сравнивает с оригиналом."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"📊 ВАЛИДАЦИЯ")
    print(f"{'='*60}")

    if best_model_path is None or not Path(best_model_path).exists():
        print("❌ best.pt не найден, валидация пропущена")
        return None, None

    metrics_new = None
    metrics_old = None

    # Валидация новой (дообученной) модели
    print(f"\n🔍 Новая модель: {best_model_path}")
    model_new = YOLO(str(best_model_path))
    val_results_new = model_new.val(
        data=config["data"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        workers=config["workers"],
        verbose=True,
    )
    metrics_new = extract_metrics(val_results_new)
    print_metrics(metrics_new, label="Новая (дообученная)")

    # Валидация оригинальной модели для сравнения
    print(f"\n🔍 Оригинальная модель: {config['model']}")
    model_old = YOLO(config["model"])
    val_results_old = model_old.val(
        data=config["data"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        workers=config["workers"],
        verbose=False,
    )
    metrics_old = extract_metrics(val_results_old)
    print_metrics(metrics_old, label="Оригинальная")

    # Сравнение
    print_comparison(metrics_old, metrics_new)

    return metrics_old, metrics_new


def extract_metrics(val_results):
    """Извлекает ключевые метрики из результатов валидации."""
    try:
        box = val_results.box
        return {
            "mAP50":    round(float(box.map50), 4),
            "mAP50-95": round(float(box.map),   4),
            "precision": round(float(box.mp),   4),
            "recall":    round(float(box.mr),   4),
        }
    except Exception as e:
        print(f"⚠️ Не удалось извлечь метрики: {e}")
        return {}


def print_metrics(metrics, label=""):
    """Выводит метрики в читаемом виде."""
    if not metrics:
        return
    print(f"\n   [{label}]")
    print(f"   mAP@0.5     : {metrics.get('mAP50',    'N/A')}")
    print(f"   mAP@0.5:0.95: {metrics.get('mAP50-95', 'N/A')}")
    print(f"   Precision   : {metrics.get('precision', 'N/A')}")
    print(f"   Recall      : {metrics.get('recall',    'N/A')}")


def print_comparison(old, new):
    """Выводит сравнительную таблицу метрик."""
    if not old or not new:
        return

    print(f"\n{'='*60}")
    print(f"📈 СРАВНЕНИЕ: оригинал → дообученная")
    print(f"{'='*60}")

    keys = [("mAP50", "mAP@0.5"), ("mAP50-95", "mAP@0.5:0.95"),
            ("precision", "Precision"), ("recall", "Recall")]

    for key, label in keys:
        o = old.get(key, 0)
        n = new.get(key, 0)
        diff = n - o
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
        sign  = "+" if diff > 0 else ""
        print(f"   {label:<16}: {o:.4f} → {n:.4f}  {arrow} {sign}{diff:.4f}")

    print(f"{'='*60}")


# ══════════════════════════════════════════════
# СОХРАНЕНИЕ ОТЧЁТА
# ══════════════════════════════════════════════

def save_report(config, best_model_path, metrics_old, metrics_new, elapsed):
    """Сохраняет JSON-отчёт с результатами обучения."""
    report = {
        "timestamp":      datetime.now().isoformat(),
        "training_time":  time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "config":         config,
        "best_model":     str(best_model_path) if best_model_path else None,
        "metrics_before": metrics_old or {},
        "metrics_after":  metrics_new or {},
    }

    report_path = Path(config["project"]) / config["name"] / "finetune_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Отчёт сохранён: {report_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO модели на данных дальнего плана")
    parser.add_argument("--model",   default=DEFAULT_CONFIG["model"],   help="Путь к базовой модели (.pt)")
    parser.add_argument("--data",    default=DEFAULT_CONFIG["data"],    help="Путь к data.yaml")
    parser.add_argument("--epochs",  default=DEFAULT_CONFIG["epochs"],  type=int)
    parser.add_argument("--imgsz",   default=DEFAULT_CONFIG["imgsz"],   type=int)
    parser.add_argument("--lr0",     default=DEFAULT_CONFIG["lr0"],     type=float)
    parser.add_argument("--freeze",  default=DEFAULT_CONFIG["freeze"],  type=int)
    parser.add_argument("--batch",   default=DEFAULT_CONFIG["batch"],   type=int)
    parser.add_argument("--device",  default=DEFAULT_CONFIG["device"],  help="CPU/GPU: '' авто, '0' GPU0, 'cpu'")
    parser.add_argument("--name",    default=DEFAULT_CONFIG["name"],    help="Имя эксперимента")
    parser.add_argument("--skip_val", action="store_true",              help="Пропустить валидацию")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    config.pop("skip_val", None)
    skip_val = args.skip_val

    # Проверка
    check_environment(config)

    # Обучение
    train_results, best_model_path, elapsed = run_training(config)

    # Валидация
    metrics_old, metrics_new = None, None
    if not skip_val:
        metrics_old, metrics_new = run_validation(best_model_path, config)

    # Отчёт
    save_report(config, best_model_path, metrics_old, metrics_new, elapsed)

    print(f"\n🎉 Готово!")
    if best_model_path and Path(best_model_path).exists():
        print(f"   Лучшая модель: {best_model_path}")
        print(f"   Замените в app.py путь на новую модель для тестирования.")


if __name__ == "__main__":
    main()
