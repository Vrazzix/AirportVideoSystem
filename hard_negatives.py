"""
Hard Negatives Pipeline

Полный автоматический пайплайн:
1. Скачивает изображения-обманки (двигатели, буквы O, конусы и т.д.)
2. Создает пустые .txt файлы меток (= "здесь НЕТ колес/колодок")
3. Копирует в train датасет
4. Проверяет баланс датасета

Использование:
    pip install requests beautifulsoup4 Pillow

    # Скачать + сразу добавить в train:
    python hard_negatives.py --train_dir "augm_split/train_augmented"

    # Только скачать (без интеграции):
    python hard_negatives.py --download_only

    # Уже скачано, только интегрировать:
    python hard_negatives.py --skip_download --train_dir "augm_split/train_augmented"
"""

import argparse
import os
import re
import time
import hashlib
import shutil
import requests
from pathlib import Path
from urllib.parse import quote_plus

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


# ==============================================================
# SEARCH QUERIES FOR HARD NEGATIVES
# ==============================================================

QUERIES = {
    "engines": [
        "aircraft engine front view close up",
        "jet engine nacelle side view",
        "airplane turbine intake close",
        "CFM56 engine front",
        "aircraft engine cowling",
        "turbofan engine front view airport",
        "airplane engine on tarmac",
        "jet engine ground level photo",
        "aircraft APU exhaust close up",
        "airplane engine maintenance close",
        "wide body aircraft engine",
        "narrow body jet engine front",
        "aircraft engine fan blades close",
        "turbine engine nacelle airport",
        "airplane engine pylon close up",
    ],
    "round_objects": [
        "airport traffic cone close up",
        "runway lights close up",
        "taxiway lights embedded",
        "airport beacon light",
        "aircraft navigation light close",
        "airplane landing light round",
        "airport manhole cover tarmac",
        "jet blast deflector round",
        "aircraft fuel cap close up",
        "airplane porthole window close",
        "aircraft window row exterior",
    ],
    "text_signs": [
        "airport apron markings text",
        "runway numbers painted ground",
        "taxiway signs airport close up",
        "airport gate number sign",
        "aircraft registration number fuselage",
        "airplane livery text close up",
        "airport terminal sign letters",
        "runway hold position sign",
        "airport safety sign close up",
        "NO ENTRY sign airport apron",
    ],
    "covers_caps": [
        "aircraft pitot tube cover",
        "airplane wheel hub cap",
        "aircraft brake disc close up",
        "airplane hydraulic reservoir cap",
        "aircraft fuel tank cap",
        "ground power unit connector round",
        "airplane antenna dome radome",
        "aircraft static port close up",
    ],
    "ground_equipment": [
        "airport GPU ground power unit",
        "baggage cart wheels airport",
        "aircraft tow bar connector",
        "pushback tractor front view",
        "airport fire extinguisher close",
        "ground handling equipment wheels",
        "belt loader wheels airport",
        "fuel truck wheels airport",
    ],
}

TOTAL_QUERIES = sum(len(v) for v in QUERIES.values())


# ==============================================================
# BING IMAGE SCRAPER
# ==============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_bing_image_urls(query, max_results=40):
    urls = []
    seen = set()

    for offset in range(0, max_results, 35):
        search_url = (
            f"https://www.bing.com/images/search"
            f"?q={quote_plus(query)}"
            f"&first={offset}&count=35"
            f"&qft=+filterui:photo-photo"
        )
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception:
            break

        if HAS_BS4:
            soup = BeautifulSoup(resp.text, "html.parser")
            for a_tag in soup.find_all("a", {"class": "iusc"}):
                m = a_tag.get("m")
                if m:
                    match = re.search(r'"murl":"(https?://[^"]+)"', m)
                    if match:
                        url = match.group(1)
                        if url not in seen:
                            seen.add(url)
                            urls.append(url)
        else:
            found = re.findall(r'"murl":"(https?://[^"]+)"', resp.text)
            for url in found:
                if url not in seen:
                    seen.add(url)
                    urls.append(url)

        if len(urls) >= max_results:
            break
        time.sleep(0.5)

    return urls[:max_results]


def download_image(url, save_path, min_size=150):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.content

        if HAS_PIL:
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(save_path, "JPEG", quality=90)
        else:
            with open(save_path, 'wb') as f:
                f.write(data)
        return True
    except Exception:
        return False


# ==============================================================
# STEP 1: DOWNLOAD
# ==============================================================

def download_hard_negatives(output_dir, max_per_query=40):
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    seen_hashes = set()
    total = 0
    category_counts = {}

    print(f"\n{'='*60}")
    print(f"  STEP 1: DOWNLOAD HARD NEGATIVES")
    print(f"  Categories: {len(QUERIES)} | Queries: {TOTAL_QUERIES} | Max/query: {max_per_query}")
    print(f"{'='*60}")

    for cat_name, queries in QUERIES.items():
        cat_count = 0
        print(f"\n  Category: {cat_name} ({len(queries)} queries)")

        for qi, query in enumerate(queries):
            print(f"    [{qi+1}/{len(queries)}] \"{query}\"", end="")

            urls = get_bing_image_urls(query, max_results=max_per_query)
            downloaded = 0

            for url in urls:
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                filename = f"neg_{cat_name}_{url_hash}.jpg"
                filepath = os.path.join(img_dir, filename)

                if os.path.exists(filepath):
                    continue

                if download_image(url, filepath):
                    with open(filepath, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                    if content_hash in seen_hashes:
                        os.remove(filepath)
                        continue
                    seen_hashes.add(content_hash)
                    downloaded += 1
                    cat_count += 1
                    total += 1

            print(f" -> {downloaded}")
            time.sleep(0.8)

        category_counts[cat_name] = cat_count
        print(f"    Subtotal {cat_name}: {cat_count}")

    print(f"\n{'='*60}")
    print(f"  Download complete: {total} images")
    for cat, cnt in category_counts.items():
        print(f"    {cat}: {cnt}")
    print(f"{'='*60}")

    return total


# ==============================================================
# STEP 2: CREATE EMPTY LABELS
# ==============================================================

def create_empty_labels(output_dir):
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    count = 0

    for f in os.listdir(img_dir):
        if Path(f).suffix.lower() in img_extensions:
            stem = Path(f).stem
            label_path = os.path.join(lbl_dir, stem + ".txt")
            open(label_path, 'w').close()
            count += 1

    print(f"\n  STEP 2: Created {count} empty label files")
    print(f"    Location: {lbl_dir}")
    return count


# ==============================================================
# STEP 3: INTEGRATE INTO TRAIN
# ==============================================================

def integrate_into_train(output_dir, train_dir):
    src_img = os.path.join(output_dir, "images")
    src_lbl = os.path.join(output_dir, "labels")
    dst_img = os.path.join(train_dir, "images")
    dst_lbl = os.path.join(train_dir, "labels")

    if not os.path.exists(dst_img):
        print(f"\n  ERROR: Train images dir not found: {dst_img}")
        print(f"    Make sure --train_dir points to folder with images/ and labels/")
        return 0

    os.makedirs(dst_lbl, exist_ok=True)
    copied = 0

    for f in os.listdir(src_img):
        src_path = os.path.join(src_img, f)
        dst_path = os.path.join(dst_img, f)

        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

            stem = Path(f).stem
            lbl_src = os.path.join(src_lbl, stem + ".txt")
            lbl_dst = os.path.join(dst_lbl, stem + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            copied += 1

    print(f"\n  STEP 3: Copied {copied} images + empty labels into train")
    print(f"    Destination: {dst_img}")

    total_train = len([
        f for f in os.listdir(dst_img)
        if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ])
    print(f"    Total images in train now: {total_train}")

    return copied


# ==============================================================
# STEP 4: VERIFY DATASET BALANCE
# ==============================================================

def verify_dataset(train_dir):
    img_dir = os.path.join(train_dir, "images")
    lbl_dir = os.path.join(train_dir, "labels")

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in img_extensions]

    missing = []
    empty_labels = 0
    non_empty_labels = 0

    for img_file in images:
        stem = Path(img_file).stem
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        if not os.path.exists(lbl_path):
            missing.append(img_file)
        else:
            if os.path.getsize(lbl_path) == 0:
                empty_labels += 1
            else:
                non_empty_labels += 1

    print(f"\n  STEP 4: Dataset verification")
    print(f"    Total images:       {len(images)}")
    print(f"    With annotations:   {non_empty_labels}")
    print(f"    Empty (negatives):  {empty_labels}")
    print(f"    Missing labels:     {len(missing)}")

    if missing:
        print(f"\n    WARNING: Images without label file:")
        for f in missing[:10]:
            print(f"      {f}")

    neg_ratio = empty_labels / len(images) * 100 if images else 0
    print(f"\n    Hard negatives ratio: {neg_ratio:.1f}%")

    if neg_ratio > 30:
        print(f"    WARNING: Too many negatives (>30%). Model may become too conservative.")
        print(f"    Recommended: 10-25%")
    elif neg_ratio < 5:
        print(f"    WARNING: Few negatives (<5%). Add more to fight false positives.")
    else:
        print(f"    OK: Good balance!")

    return len(missing) == 0


# ==============================================================
# MAIN
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Hard Negatives: download, label, integrate")
    parser.add_argument("--output_dir", default="hard_negatives", help="Folder for downloaded hard negatives")
    parser.add_argument("--train_dir", help="Train folder (with images/ and labels/) to integrate into")
    parser.add_argument("--max_per_query", type=int, default=40, help="Max images per search query")
    parser.add_argument("--download_only", action="store_true", help="Only download, do not integrate")
    parser.add_argument("--skip_download", action="store_true", help="Skip download (if already done)")

    args = parser.parse_args()

    print("""
================================================================
   HARD NEGATIVES PIPELINE

   Goal: teach the model NOT to detect:
     - Aircraft engines (round -> not wheels)
     - Letters O, Q, 0 in text
     - Traffic cones, lights, signs
     - Ground equipment
================================================================
    """)

    # Step 1
    if not args.skip_download:
        total = download_hard_negatives(args.output_dir, args.max_per_query)
        if total == 0:
            print("  ERROR: Nothing downloaded. Check internet connection.")
            return
    else:
        img_dir = os.path.join(args.output_dir, "images")
        if os.path.exists(img_dir):
            total = len(os.listdir(img_dir))
            print(f"  Download skipped. Found {total} images in {img_dir}")
        else:
            print(f"  ERROR: Folder not found: {img_dir}")
            return

    # Step 2
    create_empty_labels(args.output_dir)

    if args.download_only:
        print(f"\n  Done (download only).")
        print(f"  To integrate, run:")
        print(f'  python hard_negatives.py --skip_download --output_dir "{args.output_dir}" --train_dir "path/to/train"')
        return

    # Step 3
    if args.train_dir:
        integrate_into_train(args.output_dir, args.train_dir)
        # Step 4
        verify_dataset(args.train_dir)
    else:
        print(f"\n  --train_dir not specified. Integration skipped.")
        print(f"  Copy manually:")
        print(f"    {args.output_dir}/images/* -> train/images/")
        print(f"    {args.output_dir}/labels/* -> train/labels/")

    print(f"\n{'='*60}")
    print(f"  DONE! Next: train the model")
    print(f"  yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=1280 batch=8")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
