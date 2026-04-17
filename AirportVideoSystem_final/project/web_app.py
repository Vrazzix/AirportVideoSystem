"""
Aircraft & Human Detection System — Flask Web Application
Replaces Streamlit app.py with a proper web server.

Run:
    python web_app.py
    open http://localhost:5000
"""

import json
import os
import sys
import tempfile
import threading
import time
import uuid

import cv2
import torch
from flask import (Flask, Response, jsonify, render_template,
                   request, send_file, stream_with_context)

sys.path.insert(0, os.path.dirname(__file__))
from modules import SimpleTracker, process_frame

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024   # 2 GB

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'flask_output')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'uploaded_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Allowed model extensions (single-file formats)
_MODEL_EXTS = {'.pt', '.onnx', '.engine', '.tflite', '.pb'}

# job_id → job dict
_jobs: dict = {}
_jobs_lock  = threading.Lock()

# YOLO model cache (path → YOLO instance)
_model_cache: dict = {}
_model_lock  = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_from_path(path: str) -> str:
    """Detect model format label from path."""
    p = path.rstrip('/\\')
    if os.path.isdir(p):
        return 'openvino'
    ext = os.path.splitext(p)[1].lower()
    return {'.pt': 'pytorch', '.onnx': 'onnx',
            '.engine': 'tensorrt', '.tflite': 'tflite',
            '.pb': 'tensorflow'}.get(ext, 'unknown')


def _list_models() -> list:
    """
    Return all available models: uploaded ones + defaults that still exist.
    Each entry: {name, path, format, size_mb}
    """
    found = {}   # path → entry (dedup)

    # 1. Default model dirs to scan
    default_dirs = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'models'),
        os.path.join(os.path.dirname(__file__), 'models'),
    ]
    for d in default_dirs:
        d = os.path.normpath(d)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in _MODEL_EXTS and os.path.isfile(fp):
                found[fp] = _model_entry(fp)
            # OpenVINO IR directories
            if os.path.isdir(fp) and fn.endswith('_openvino_model'):
                found[fp] = _model_entry(fp)

    # 2. Uploaded models dir
    if os.path.isdir(MODELS_DIR):
        for fn in os.listdir(MODELS_DIR):
            fp = os.path.join(MODELS_DIR, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in _MODEL_EXTS and os.path.isfile(fp):
                found[fp] = _model_entry(fp)
            if os.path.isdir(fp) and fn.endswith('_openvino_model'):
                found[fp] = _model_entry(fp)

    return sorted(found.values(), key=lambda x: x['name'])


def _model_entry(path: str) -> dict:
    if os.path.isdir(path):
        size = sum(os.path.getsize(os.path.join(path, f))
                   for f in os.listdir(path)
                   if os.path.isfile(os.path.join(path, f)))
    else:
        size = os.path.getsize(path)
    return {
        'name':    os.path.basename(path),
        'path':    path,
        'format':  _fmt_from_path(path),
        'size_mb': round(size / 1024 / 1024, 1),
    }


def _load_model(path: str):
    if not path or not os.path.exists(path):
        return None
    with _model_lock:
        if path not in _model_cache:
            from ultralytics import YOLO
            _model_cache[path] = YOLO(path)
        return _model_cache[path]


def _new_job() -> dict:
    return {
        'status':       'pending',   # pending | processing | cancelled | done | error
        'frame':        0,
        'total':        0,
        'processed':    0,
        'fps_proc':     0.0,
        'eta':          0,
        'frame_jpg':    None,        # bytes — latest annotated frame (JPEG)
        'result':       None,
        'error':        None,
        'cancel_event': threading.Event(),   # set() to request cancellation
    }


# ─────────────────────────────────────────────────────────────────────────────
# Video processing (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _is_rtsp(path: str) -> bool:
    """True if input is a live stream URL, not a local file."""
    return path.lower().startswith(('rtsp://', 'rtsps://', 'rtmp://', 'http://', 'https://'))


def _open_capture(source: str, reconnect: bool = False) -> cv2.VideoCapture:
    """Open VideoCapture, for RTSP set buffer size to 1 to reduce latency."""
    if _is_rtsp(source):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # минимальный буфер (меньше задержки)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def _process_job(job_id: str, input_path: str, cfg: dict):
    job = _jobs[job_id]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    is_live = _is_rtsp(input_path)

    try:
        # ── Load models ───────────────────────────────────────────────────────
        loaded = {}
        if cfg['use_wheels']:
            m = _load_model(cfg['model_wheels']); m and loaded.update(combo=m)
        if cfg['use_person']:
            m = _load_model(cfg['model_person']); m and loaded.update(person=m)
        if cfg['use_pose']:
            m = _load_model(cfg['model_pose']);   m and loaded.update(pose=m)

        if not loaded:
            raise RuntimeError('No models loaded — check model paths')

        # ── Video info ────────────────────────────────────────────────────────
        cap = _open_capture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open source: {input_path}')

        total_frames = -1 if is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        job['total'] = total_frames   # -1 = live stream (unknown)

        # ── Tracker ───────────────────────────────────────────────────────────
        tracker = (SimpleTracker(iou_threshold=cfg['iou_thresh'])
                   if cfg['tracking_enabled'] else None)

        filter_cfg = {
            'filter_enabled':   cfg['filter_enabled'],
            'wheel_class_id':   cfg['wheel_class'],
            'chock_class_id':   cfg['chock_class'],
            'wheel_min_area':   cfg['wheel_min_area'],
            'wheel_max_aspect': cfg['wheel_max_asp'],
            'wheel_min_aspect': cfg['wheel_min_asp'],
            'use_zone_filter':  False,
            'zone_pct':         70,
            'chock_min_area':   cfg['chock_min_area'],
            'chock_max_area':   cfg['chock_max_area'],
            'chock_max_aspect': cfg['chock_max_asp'],
        }

        # ── Writer (только для файлов, не для live-потока) ────────────────────
        raw_out = os.path.join(job_dir, 'raw.mp4')
        out_fps = max(fps / cfg['every_n'], 1.0)
        writer  = None if is_live else cv2.VideoWriter(
            raw_out, cv2.VideoWriter_fourcc(*'mp4v'),
            out_fps, (vid_w, vid_h))

        # ── Main loop ─────────────────────────────────────────────────────────
        cap              = _open_capture(input_path)
        reconnect_delay  = 2       # секунды между попытками переподключения
        max_reconnects   = 10      # максимум попыток для live-потока
        reconnect_count  = 0
        frame_idx        = 0
        proc_cnt      = 0
        svc_status    = 'ОЖИДАНИЕ: Установите колодки'
        svc_color     = (0, 0, 255)
        t_start       = time.time()
        cancel_event  = job['cancel_event']
        job['status'] = 'processing'

        while cap.isOpened():
            # ── Cancellation check ────────────────────────────────────────────
            if cancel_event.is_set():
                job['status'] = 'cancelled'
                break

            ret, frame = cap.read()
            if not ret:
                if not is_live:
                    break   # файл закончился — нормальный выход
                # RTSP: потеря соединения — пытаемся переподключиться
                reconnect_count += 1
                if reconnect_count > max_reconnects:
                    raise RuntimeError(
                        f'RTSP stream lost after {max_reconnects} reconnect attempts')
                job['status'] = 'reconnecting'
                cap.release()
                time.sleep(reconnect_delay)
                cap = _open_capture(input_path)
                if cap.isOpened():
                    job['status'] = 'processing'
                    reconnect_count = 0
                continue

            if frame_idx % cfg['every_n'] == 0:
                annotated, svc_status, svc_color = process_frame(
                    frame,
                    loaded.get('combo'), loaded.get('person'), loaded.get('pose'),
                    cfg['use_wheels'], cfg['use_person'], cfg['use_pose'],
                    cfg['conf_wheels'], cfg['conf_person'], cfg['conf_pose_kpt'],
                    cfg['imgsz'],
                    cfg['line_thickness'], cfg['font_scale'],
                    svc_status, svc_color, cfg['show_status_bar'],
                    tracker, cfg['show_track_ids'], True,
                    filter_cfg=filter_cfg,
                    tracking_enabled=cfg['tracking_enabled'],
                )
                if writer is not None:
                    writer.write(annotated)
                proc_cnt += 1

                # Store latest preview frame every 10 processed frames
                if proc_cnt % 10 == 0:
                    _, buf = cv2.imencode('.jpg', annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                    job['frame_jpg'] = buf.tobytes()

            # Update progress
            elapsed  = time.time() - t_start
            fps_proc = proc_cnt / elapsed if elapsed > 0 else 0
            if is_live:
                eta = -1   # живой поток — ETA неизвестен
            else:
                eta = int((total_frames - frame_idx) / max(fps_proc * cfg['every_n'], 0.01))
            job.update(frame=frame_idx + 1, processed=proc_cnt,
                       fps_proc=round(fps_proc, 1), eta=eta)
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        total_time = time.time() - t_start

        # ── Cancelled — clean up partial files and exit ───────────────────────
        if job['status'] == 'cancelled':
            if not is_live:
                for p in [raw_out]:
                    try:
                        if os.path.exists(p): os.unlink(p)
                    except OSError:
                        pass
            return

        # ── Re-encode H.264 (только для файлов) ───────────────────────────────
        final_video = None
        if not is_live and writer is not None:
            final_video = os.path.join(job_dir, 'result.mp4')
            ret_code = os.system(
                f'ffmpeg -y -i "{raw_out}" -vcodec libx264 -preset fast '
                f'"{final_video}" -loglevel error')
            if ret_code != 0 or not os.path.exists(final_video) or os.path.getsize(final_video) == 0:
                os.rename(raw_out, final_video)
            elif os.path.exists(raw_out):
                os.unlink(raw_out)

        job['result'] = {
            'video_path': final_video,
            'timing': {
                'total_sec':  round(total_time, 1),
                'frames_out': proc_cnt,
                'avg_fps':    round(proc_cnt / total_time, 1) if total_time > 0 else 0,
            },
        }
        job['status'] = 'done'

    except Exception as exc:
        import traceback
        job['error']  = str(exc)
        job['status'] = 'error'
        print(f'[JOB {job_id}] ERROR:', traceback.format_exc())
    finally:
        if not is_live:   # файл — удаляем временный
            try:
                os.unlink(input_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           has_cuda=torch.cuda.is_available())


@app.route('/rtsp/connect', methods=['POST'])
def rtsp_connect():
    """
    Принять RTSP URL и создать job для обработки живого потока.
    Body JSON: { "url": "rtsp://user:pass@192.168.1.10:554/stream1" }
    """
    data = request.get_json(force=True)
    rtsp_url = (data or {}).get('url', '').strip()
    if not rtsp_url:
        return jsonify(error='url is required'), 400
    if not _is_rtsp(rtsp_url):
        return jsonify(error='Not a valid stream URL (rtsp://, rtmp://, http://)'), 400

    # Проверяем что камера доступна
    cap = _open_capture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        return jsonify(error=f'Cannot connect to stream: {rtsp_url}'), 502

    ret, _ = cap.read()
    cap.release()
    if not ret:
        return jsonify(error='Stream opened but no frames received'), 502

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        job = _new_job()
        job['input_path'] = rtsp_url
        job['is_live']    = True
        _jobs[job_id] = job

    return jsonify(
        job_id=job_id,
        meta=dict(url=rtsp_url, live=True, fps=0, frames=-1)
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify(error='No file'), 400

    f = request.files['video']
    if not f.filename:
        return jsonify(error='Empty filename'), 400

    suffix = os.path.splitext(f.filename)[1] or '.mp4'
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()

    # Read video metadata
    cap   = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        job = _new_job()
        job['input_path'] = tmp.name
        _jobs[job_id] = job

    return jsonify(
        job_id=job_id,
        meta=dict(width=w, height=h, fps=round(fps, 2),
                  frames=total, duration=round(total / fps if fps > 0 else 0, 1))
    )


@app.route('/process/<job_id>', methods=['POST'])
def start_process(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify(error='Job not found'), 404
    if job['status'] not in ('pending',):
        return jsonify(error='Job already started'), 409

    cfg = request.get_json(force=True)

    # Defaults for any missing keys
    cfg.setdefault('model_wheels',    'C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestBoots_v2.pt')
    cfg.setdefault('model_person',    'C:\\Users\\shche\\Desktop\\Application_for_models\\models\\person.pt')
    cfg.setdefault('model_pose',      'C:\\Users\\shche\\Desktop\\Application_for_models\\models\\BestPose.pt')
    cfg.setdefault('use_wheels',      True)
    cfg.setdefault('use_person',      True)
    cfg.setdefault('use_pose',        True)
    cfg.setdefault('conf_wheels',     0.15)
    cfg.setdefault('conf_person',     0.50)
    cfg.setdefault('conf_pose_kpt',   0.50)
    cfg.setdefault('imgsz',           1280)
    cfg.setdefault('every_n',         1)
    cfg.setdefault('wheel_class',     1)
    cfg.setdefault('chock_class',     0)
    cfg.setdefault('filter_enabled',  True)
    cfg.setdefault('wheel_min_area',  3000)
    cfg.setdefault('wheel_max_asp',   2.5)
    cfg.setdefault('wheel_min_asp',   0.4)
    cfg.setdefault('chock_min_area',  1000)
    cfg.setdefault('chock_max_area',  40000)
    cfg.setdefault('chock_max_asp',   4.0)
    cfg.setdefault('tracking_enabled', True)
    cfg.setdefault('iou_thresh',       0.3)
    cfg.setdefault('line_thickness',   2)
    cfg.setdefault('font_scale',       0.6)
    cfg.setdefault('show_status_bar',  True)
    cfg.setdefault('show_track_ids',   True)

    t = threading.Thread(
        target=_process_job,
        args=(job_id, job['input_path'], cfg),
        daemon=True,
    )
    t.start()
    return jsonify(ok=True)


@app.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify(error='Job not found'), 404
    if job['status'] not in ('pending', 'processing'):
        return jsonify(error='Job is not running'), 409
    job['cancel_event'].set()
    return jsonify(ok=True)


@app.route('/stream/<job_id>')
def stream(job_id: str):
    """SSE stream: sends progress updates until job is done or errored."""
    def generate():
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                yield f'data: {json.dumps({"error": "not found"})}\n\n'
                return

            payload = {
                'status':    job['status'],
                'frame':     job['frame'],
                'total':     job['total'],
                'processed': job['processed'],
                'fps_proc':  job['fps_proc'],
                'eta':       job['eta'],
            }

            if job['status'] == 'done':
                r = job['result']
                payload['result'] = {
                    'has_video': r['video_path'] is not None,
                    'timing':    r['timing'],
                }
                yield f'data: {json.dumps(payload)}\n\n'
                return
            elif job['status'] == 'cancelled':
                yield f'data: {json.dumps(payload)}\n\n'
                return
            elif job['status'] == 'error':
                payload['error'] = job['error']
                yield f'data: {json.dumps(payload)}\n\n'
                return
            else:
                yield f'data: {json.dumps(payload)}\n\n'
                time.sleep(1.0)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/frame/<job_id>')
def latest_frame(job_id: str):
    """Returns the latest annotated frame as JPEG (for live preview)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job['frame_jpg'] is None:
        # Return 1×1 transparent placeholder
        import base64
        px = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk'
            'YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')
        return Response(px, mimetype='image/png')
    return Response(job['frame_jpg'], mimetype='image/jpeg')


@app.route('/download/<job_id>/video')
def download_video(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job['result'] is None:
        return 'Not found', 404
    path = job['result']['video_path']
    if not path or not os.path.exists(path):
        return 'File not found', 404
    return send_file(path, as_attachment=True,
                     download_name='processed_video.mp4',
                     mimetype='video/mp4')


@app.route('/video/<job_id>')
def stream_video(job_id: str):
    """Serve result video with Range support for HTML5 player."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job['result'] is None:
        return 'Not found', 404
    path = job['result']['video_path']
    if not path or not os.path.exists(path):
        return 'Not found', 404
    return send_file(path, mimetype='video/mp4',
                     conditional=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model management routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/models/list')
def models_list():
    return jsonify(models=_list_models())


@app.route('/models/upload', methods=['POST'])
def models_upload():
    """
    Upload one or more model files.
    Single-file formats: .pt  .onnx  .engine  .tflite  .pb
    OpenVINO IR: upload .xml + .bin together (multipart files[]).
    """
    files = request.files.getlist('files')
    if not files:
        return jsonify(error='No files provided'), 400

    saved = []
    errors = []

    # Check if this is an OpenVINO pair (.xml + .bin)
    exts = {os.path.splitext(f.filename)[1].lower() for f in files}
    is_openvino = '.xml' in exts and '.bin' in exts

    if is_openvino:
        # Determine base name from .xml file
        xml_file = next(f for f in files
                        if os.path.splitext(f.filename)[1].lower() == '.xml')
        base = os.path.splitext(xml_file.filename)[0]
        ov_dir = os.path.join(MODELS_DIR, base + '_openvino_model')
        os.makedirs(ov_dir, exist_ok=True)

        for f in files:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext in ('.xml', '.bin'):
                dest = os.path.join(ov_dir, f.filename)
                f.save(dest)
                saved.append(f.filename)

        # Evict cached model if exists
        with _model_lock:
            _model_cache.pop(ov_dir, None)

        saved_entry = _model_entry(ov_dir)
        return jsonify(saved=[saved_entry])

    # Single-file uploads
    for f in files:
        fn  = f.filename
        ext = os.path.splitext(fn)[1].lower()
        if ext not in _MODEL_EXTS:
            errors.append(f'{fn}: unsupported format (allowed: '
                          f'{", ".join(_MODEL_EXTS)})')
            continue

        dest = os.path.join(MODELS_DIR, fn)
        f.save(dest)

        # Evict old cache entry
        with _model_lock:
            _model_cache.pop(dest, None)

        saved.append(_model_entry(dest))

    if errors and not saved:
        return jsonify(error='; '.join(errors)), 400
    return jsonify(saved=saved, errors=errors)


@app.route('/models/delete', methods=['POST'])
def models_delete():
    """Delete an uploaded model by path (only files inside MODELS_DIR)."""
    data = request.get_json(force=True)
    path = data.get('path', '')

    # Security: only allow deleting from MODELS_DIR
    norm_path    = os.path.normpath(path)
    norm_models  = os.path.normpath(MODELS_DIR)
    if not norm_path.startswith(norm_models):
        return jsonify(error='Cannot delete files outside upload directory'), 403

    if not os.path.exists(norm_path):
        return jsonify(error='File not found'), 404

    try:
        if os.path.isdir(norm_path):
            import shutil
            shutil.rmtree(norm_path)
        else:
            os.unlink(norm_path)
        with _model_lock:
            _model_cache.pop(norm_path, None)
        return jsonify(ok=True)
    except OSError as e:
        return jsonify(error=str(e)), 500


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 55)
    print('  Aircraft & Human Detection — Flask App')
    print(f'  Device : {"CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"}')
    print('  URL    : http://localhost:5000')
    print('=' * 55)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
