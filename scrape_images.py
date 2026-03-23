"""
🔍 Парсер изображений колодок и колёс самолёта

Скачивает изображения из Bing Image Search по набору запросов.
Не требует API-ключа — использует публичный поиск.

Использование:
    pip install requests beautifulsoup4 Pillow
    python scrape_images.py --output_dir downloaded_images --max_per_query 80

После скачивания:
    1. Вручную удалите нерелевантные изображения
    2. Разметьте в LabelImg / CVAT / Roboflow
    3. Запустите split_dataset.py и augmentation.py
"""

import argparse
import os
import re
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ══════════════════════════════════════════════
# ПОИСКОВЫЕ ЗАПРОСЫ
# ══════════════════════════════════════════════

# Основные запросы для колодок (chocks)
CHOCK_QUERIES = [
    # Английские
    "aircraft wheel chocks",
    "airplane chocks on wheels",
    "aircraft chocks placement",
    "airplane wheel chocks close up",
    "aviation wheel chocks airport",
    "aircraft chocks ground handling",
    "plane wheel chocks tarmac",
    "rubber aircraft chocks",
    "yellow aircraft wheel chocks",
    "orange aviation chocks",
    "aircraft chocks and cones",
    "ground crew placing chocks",
    "aircraft chocks removal",
    "nose gear chocks aircraft",
    "main gear chocks airplane",
    "wooden aircraft wheel chocks",
    "aircraft parking chocks",
    "chocking aircraft procedure",
    "aircraft chocks safety",
    "airline ground handling chocks",
    # С конкретными типами самолётов
    "boeing 737 wheel chocks",
    "airbus a320 wheel chocks",
    "boeing 777 chocks",
    "wide body aircraft chocks",
    "narrow body airplane chocks",
    "commercial aircraft chocks ground",
    # Разные ракурсы
    "aircraft chocks front view",
    "aircraft chocks side view",
    "chocks under airplane tire",
    "aircraft tire with chocks close",
]

# Дальний план (приоритетные — скачиваются первыми)
CHOCK_DISTANT_QUERIES = [
    "airplane parked gate chocks distant",
    "aircraft apron parking chocks far view",
    "airport ramp aircraft chocks wide angle",
    "airplane parked terminal with chocks",
    "aircraft on apron chocks full body",
    "wide angle aircraft parking chocks",
    "aircraft wheel chocks from distance",
    "airplane full view parked chocks",
    "airport apron overview aircraft chocks",
    "aircraft stand chocks aerial view",
    "ground crew aircraft chocks wide shot",
    "plane parked airport chocks overview",
]

# Основные запросы для колёс (wheels)
WHEEL_QUERIES = [
    "aircraft landing gear wheels",
    "airplane wheels close up",
    "aircraft main gear tires",
    "airplane nose gear wheel",
    "aircraft tire tarmac",
    "airplane landing gear on ground",
    "commercial aircraft wheels",
    "boeing landing gear close up",
    "airbus landing gear wheels",
    "aircraft wheel brake assembly",
    "airplane tire ground level",
    "aircraft gear on apron",
    "plane wheels parked",
    "aircraft main wheel assembly",
    "airplane tire runway",
]

# Комбинированные (колёса + колодки вместе)
COMBO_QUERIES = [
    "aircraft wheel chocks placed",
    "airplane chocked and parked",
    "ground handling chocks wheels",
    "aircraft chocks under wheels photo",
    "airplane parked with chocks",
    "aircraft ground service chocks",
    "ramp operations chocks aircraft",
    "airline chocks safety procedure photo",
]

# Hard negatives (круглые объекты, НЕ колёса — для борьбы с FP)
HARD_NEGATIVE_QUERIES = [
    "aircraft engine front view",
    "jet engine nacelle",
    "airplane turbine close up",
    "airport traffic cone",
    "aircraft navigation light",
    "airplane window porthole",
]


# ══════════════════════════════════════════════
# BING IMAGE SCRAPER
# ══════════════════════════════════════════════

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_bing_image_urls(query, max_results=80):
    """
    Извлекает URL изображений из Bing Image Search.
    """
    urls = []
    seen = set()

    for offset in range(0, max_results, 35):
        search_url = (
            f"https://www.bing.com/images/search"
            f"?q={quote_plus(query)}"
            f"&first={offset}"
            f"&count=35"
            f"&qft=+filterui:photo-photo"  # Только фотографии
        )

        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"      ⚠️ Ошибка запроса: {e}")
            break

        if HAS_BS4:
            soup = BeautifulSoup(resp.text, "html.parser")

            # Метод 1: через атрибут m (JSON с URL)
            for a_tag in soup.find_all("a", {"class": "iusc"}):
                m = a_tag.get("m")
                if m:
                    # Извлекаем murl из JSON-строки
                    match = re.search(r'"murl":"(https?://[^"]+)"', m)
                    if match:
                        url = match.group(1)
                        if url not in seen:
                            seen.add(url)
                            urls.append(url)

            # Метод 2: через img src (превью, менее качественные)
            if len(urls) < offset + 10:
                for img in soup.find_all("img"):
                    src = img.get("src", "") or img.get("data-src", "")
                    if src.startswith("http") and "bing.com" not in src:
                        if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                            if src not in seen:
                                seen.add(src)
                                urls.append(src)
        else:
            # Без BeautifulSoup — regex fallback
            found = re.findall(r'"murl":"(https?://[^"]+)"', resp.text)
            for url in found:
                if url not in seen:
                    seen.add(url)
                    urls.append(url)

        if len(urls) >= max_results:
            break

        time.sleep(0.5)  # Пауза между страницами

    return urls[:max_results]


def download_image(url, save_path, min_size=100, max_size_mb=15):
    """
    Скачивает изображение, проверяет валидность и размер.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        resp.raise_for_status()

        content_length = int(resp.headers.get('content-length', 0))
        if content_length > max_size_mb * 1024 * 1024:
            return False, "too large"

        data = resp.content

        if HAS_PIL:
            img = Image.open(io.BytesIO(data))
            w, h = img.size

            # Фильтр: слишком маленькие / иконки
            if w < min_size or h < min_size:
                return False, f"too small ({w}x{h})"

            # Конвертируем в RGB и сохраняем как JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(save_path, "JPEG", quality=90)
        else:
            with open(save_path, 'wb') as f:
                f.write(data)

        return True, "ok"

    except Exception as e:
        return False, str(e)


def scrape_category(queries, output_dir, category_name, max_per_query=80):
    """Скачивает изображения по списку запросов в одну папку."""

    os.makedirs(output_dir, exist_ok=True)
    total_downloaded = 0
    total_skipped = 0
    seen_hashes = set()

    print(f"\n{'='*60}")
    print(f"📂 Категория: {category_name}")
    print(f"   Запросов: {len(queries)}")
    print(f"   Макс. на запрос: {max_per_query}")
    print(f"{'='*60}")

    for qi, query in enumerate(queries):
        print(f"\n   🔍 [{qi+1}/{len(queries)}] \"{query}\"")

        urls = get_bing_image_urls(query, max_results=max_per_query)
        print(f"      Найдено URL: {len(urls)}")

        downloaded = 0
        for url in urls:
            # Генерируем уникальное имя на основе хеша URL
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = f"{category_name}_{url_hash}.jpg"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                continue

            success, reason = download_image(url, filepath)

            if success:
                # Проверка дубликатов по хешу содержимого
                with open(filepath, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()

                if content_hash in seen_hashes:
                    os.remove(filepath)
                    total_skipped += 1
                    continue

                seen_hashes.add(content_hash)
                downloaded += 1
                total_downloaded += 1
            else:
                total_skipped += 1

        print(f"      ✅ Скачано: {downloaded} | ⏭️ Пропущено: {total_skipped}")

        # Пауза между запросами
        time.sleep(1.0)

    print(f"\n   📊 Итого {category_name}: {total_downloaded} изображений")
    return total_downloaded


def main(output_dir, max_per_query=80, skip_negatives=False):
    """Основной пайплайн скачивания."""

    if not HAS_BS4:
        print("⚠️  beautifulsoup4 не установлен — будет использован regex (менее надёжно)")
        print("    pip install beautifulsoup4")

    if not HAS_PIL:
        print("⚠️  Pillow не установлен — проверка размера изображений отключена")
        print("    pip install Pillow")

    print(f"\n🚀 Начинаем сбор изображений")
    print(f"   Папка: {output_dir}")
    print(f"   Макс. на запрос: {max_per_query}")

    total = 0

    # 1. Колодки дальнего плана (первыми — приоритет)
    chock_distant_dir = os.path.join(output_dir, "chocks_distant")
    total += scrape_category(CHOCK_DISTANT_QUERIES, chock_distant_dir, "chock_distant", max_per_query)

    # 2. Колодки (все остальные запросы)
    chock_dir = os.path.join(output_dir, "chocks")
    total += scrape_category(CHOCK_QUERIES, chock_dir, "chock", max_per_query)

    # 2. Колёса
    wheel_dir = os.path.join(output_dir, "wheels")
    total += scrape_category(WHEEL_QUERIES, wheel_dir, "wheel", max_per_query)

    # 3. Комбинированные
    combo_dir = os.path.join(output_dir, "combo")
    total += scrape_category(COMBO_QUERIES, combo_dir, "combo", max_per_query)

    # 4. Hard negatives (опционально)
    if not skip_negatives:
        neg_dir = os.path.join(output_dir, "hard_negatives")
        total += scrape_category(HARD_NEGATIVE_QUERIES, neg_dir, "neg", max_per_query)

    print(f"\n{'='*60}")
    print(f"🎉 ГОТОВО! Всего скачано: {total} изображений")
    print(f"📁 Результат: {output_dir}/")
    print(f"   ├── chocks_distant/  — колодки дальнего плана (приоритет)")
    print(f"   ├── chocks/          — колодки")
    print(f"   ├── wheels/          — колёса")
    print(f"   ├── combo/           — колёса + колодки вместе")
    if not skip_negatives:
        print(f"   └── hard_negatives/  — FP-примеры (двигатели и т.д.)")
    print(f"\n📋 Следующие шаги:")
    print(f"   1. Просмотрите и УДАЛИТЕ нерелевантные фото")
    print(f"   2. Объедините все папки в одну (кроме hard_negatives)")
    print(f"   3. Разметьте в LabelImg / CVAT / Roboflow:")
    print(f"      - Класс 0: chock (колодка)")
    print(f"      - Класс 1: wheel (колесо)")
    print(f"   4. Для hard_negatives создайте ПУСТЫЕ .txt файлы")
    print(f"   5. Запустите split_dataset.py → augmentation.py → обучение")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape aircraft chock/wheel images from Bing")
    parser.add_argument("--output_dir", default="downloaded_images", help="Папка для результата")
    parser.add_argument("--max_per_query", type=int, default=80, help="Макс. изображений на запрос")
    parser.add_argument("--skip_negatives", action="store_true", help="Пропустить hard negatives")
    args = parser.parse_args()

    main(args.output_dir, args.max_per_query, args.skip_negatives)
