"""
Aircraft & Human Detection System v4
- YOLO detection: wheels, chocks, people
- Pose estimation (MPII 16 keypoints)
- IoU-based object tracking
- Semantic SLAM with 3D point cloud (Plotly)
"""

import io
import os
import tempfile
import time

import cv2
import streamlit as st
import torch

from modules import (
    SimpleTracker,
    SemanticSLAM,
    build_slam_plotly,
    process_frame,
)
from ultralytics import YOLO

_USE_HALF = torch.cuda.is_available()

# ──────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft & Human Detection",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-family: 'Inter', sans-serif; font-size: 1rem; color: #6b7280; margin-bottom: 2rem; }
    .model-card { background: linear-gradient(145deg, #f8fafc, #e2e8f0); border-radius: 12px;
                  padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid; }
    .model-card-wheels { border-left-color: #f59e0b; }
    .model-card-person { border-left-color: #3b82f6; }
    .model-card-pose   { border-left-color: #10b981; }
    .status-ready   { color: #10b981; font-weight: 600; }
    .status-missing { color: #ef4444; font-weight: 600; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #1e293b); }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] label { color: #cbd5e1 !important; }
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">✈️ Aircraft & Human Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Детекция колёс/колодок, людей и позы · Фильтрация FP · Трекинг · Semantic SLAM 3D</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Модели")
    model_path_wheels = st.text_input("🛞 Комбо-модель (колёса+колодки)", value="models/bestBoots_v2.pt")
    model_path_person = st.text_input("🧑 Модель людей", value="yolov8n.pt")
    model_path_pose   = st.text_input("🦴 Модель позы (MPII 16)", value="models/BestPose.pt")

    st.markdown("---")
    st.markdown("## 🏷️ ID классов")
    wheel_class_id = st.number_input("ID «Колесо»",   0, 20, 1)
    chock_class_id = st.number_input("ID «Колодка»",  0, 20, 0)

    st.markdown("---")
    st.markdown("## 🎛️ Пороги")
    conf_wheels    = st.slider("Conf — колёса/колодки", 0.05, 1.0, 0.15, 0.05)
    conf_person    = st.slider("Conf — люди",            0.1,  1.0, 0.5,  0.05)
    conf_pose_kpt  = st.slider("Conf — ключевые точки", 0.1,  1.0, 0.5,  0.05)
    combo_imgsz    = st.selectbox("Размер инференса комбо", [640, 960, 1280], index=2)
    process_every_n = st.slider("Каждый N-й кадр", 1, 10, 1)

    st.markdown("---")
    st.markdown("## 🔍 Фильтрация ложных срабатываний")
    filter_enabled = st.checkbox("Включить фильтрацию FP", value=True)

    st.markdown("**Колесо (Wheel):**")
    wheel_min_area   = st.slider("Мин. площадь Wheel (px²)", 500, 20000, 3000, 500,
                                 help="Отсекает мелкие ложные детекции")
    wheel_max_aspect = st.slider("Макс. aspect ratio Wheel", 1.0, 5.0, 2.5, 0.1,
                                 help="Колесо ~круглое (≈1.0)")
    wheel_min_aspect = st.slider("Мин. aspect ratio Wheel", 0.1, 1.0, 0.4, 0.05)

    st.markdown("**Колодка (Chock):**")
    chock_min_area   = st.slider("Мин. площадь Chock (px²)", 200, 10000, 1000, 200)
    chock_max_area   = st.slider("Макс. площадь Chock (px²)", 5000, 200000, 40000, 5000,
                                 help="Отсекает крупные объекты (машины и т.п.)")
    chock_max_aspect = st.slider("Макс. aspect ratio Chock", 1.0, 8.0, 4.0, 0.5)

    st.markdown("**Зона кадра:**")
    use_zone_filter = st.checkbox("Колёса только в нижних N% кадра", value=False)
    zone_pct        = st.slider("Нижний % кадра для колёс", 30, 100, 70, 5)

    st.markdown("---")
    st.markdown("## 🔗 Трекинг объектов")
    tracking_enabled  = st.checkbox("Включить трекинг (IoU-based)", value=True)
    iou_track_thresh  = st.slider("IoU порог для трекинга", 0.1, 0.8, 0.3, 0.05)

    st.markdown("---")
    st.markdown("## 🗺️ Semantic SLAM")
    slam_enabled      = st.checkbox("Включить Semantic SLAM", value=True)
    slam_min_obs      = st.slider("Мин. наблюдений для верификации", 2, 15, 3)
    slam_min_feat     = st.slider("Мин. признаков в bbox", 0, 10, 2)
    show_slam_overlay = st.checkbox("Показывать статистику SLAM", value=True)

    st.markdown("---")
    st.markdown("## 🎨 Активные модели")
    use_wheels = st.checkbox("🛞 Колёса и колодки", value=True)
    use_person = st.checkbox("🧑 Детекция людей",   value=True)
    use_pose   = st.checkbox("🦴 Поза человека",    value=True)

    st.markdown("---")
    st.markdown("## 📊 Визуализация")
    line_thickness   = st.slider("Толщина линий", 1, 5, 2)
    font_scale       = st.slider("Размер шрифта", 0.3, 1.5, 0.6, step=0.1)
    show_status_bar  = st.checkbox("Статус-бар (логика колодок)", value=True)
    show_track_ids   = st.checkbox("Показывать ID трекинга", value=True)
    show_filt_stats  = st.checkbox("Показывать счётчик отфильтрованных", value=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)


def check_model_status(path: str) -> tuple:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return "ready", f"✅ ({size_mb:.1f} MB)"
    return "missing", f"❌ `{path}`"


st.markdown("### 📦 Статус моделей")
mc1, mc2, mc3 = st.columns(3)
for name, path, css, enabled, col in [
    ("🛞 Колёса+колодки", model_path_wheels, "model-card-wheels", use_wheels, mc1),
    ("🧑 Люди",           model_path_person, "model-card-person", use_person, mc2),
    ("🦴 Поза MPII-16",   model_path_pose,   "model-card-pose",   use_pose,   mc3),
]:
    s, msg = check_model_status(path)
    sc = "status-ready" if s == "ready" else "status-missing"
    with col:
        st.markdown(
            f'<div class="model-card {css}"><strong>{name}</strong><br>'
            f'<span class="{sc}">{msg}</span><br>'
            f'<small>{"Вкл" if enabled else "Выкл"}</small></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Filter config dict (passed to process_frame / filter_detection)
# ──────────────────────────────────────────────────────────────────────────────
filter_cfg = {
    "filter_enabled":  filter_enabled,
    "wheel_class_id":  wheel_class_id,
    "chock_class_id":  chock_class_id,
    "wheel_min_area":  wheel_min_area,
    "wheel_max_aspect": wheel_max_aspect,
    "wheel_min_aspect": wheel_min_aspect,
    "use_zone_filter": use_zone_filter,
    "zone_pct":        zone_pct,
    "chock_min_area":  chock_min_area,
    "chock_max_area":  chock_max_area,
    "chock_max_aspect": chock_max_aspect,
}

# ──────────────────────────────────────────────────────────────────────────────
# Video upload & processing
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 🎬 Загрузка видео")

uploaded_file = st.file_uploader(
    "Перетащите видеофайл или нажмите для выбора",
    type=["mp4", "avi", "mov", "mkv", "wmv"],
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps if fps > 0 else 0
    cap.release()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📐 Разрешение",   f"{vid_w}×{vid_h}")
    c2.metric("🎞️ Кадров",       f"{total_frames}")
    c3.metric("⏱️ Длительность", f"{duration:.1f} сек")
    c4.metric("🔄 FPS",          f"{fps:.1f}")

    st.markdown("---")
    with st.expander("👁️ Предпросмотр", expanded=False):
        st.video(input_path)

    if st.button("🚀 Запустить обработку", type="primary", use_container_width=True):

        loaded = {}
        with st.spinner("Загрузка моделей..."):
            if use_wheels:
                m = load_model(model_path_wheels)
                if m:  loaded["combo"]  = m
                else:  st.warning(f"⚠️ {model_path_wheels}")
            if use_person:
                m = load_model(model_path_person)
                if m:  loaded["person"] = m
                else:  st.warning(f"⚠️ {model_path_person}")
            if use_pose:
                m = load_model(model_path_pose)
                if m:  loaded["pose"]   = m
                else:  st.warning(f"⚠️ {model_path_pose}")

        if not loaded:
            st.error("❌ Ни одна модель не загружена.")
        else:
            st.success(f"✅ Загружено: {', '.join(loaded.keys())}")

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            cap = cv2.VideoCapture(input_path)
            out_fps = fps / process_every_n
            writer  = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                out_fps, (vid_w, vid_h))

            progress_bar  = st.progress(0, text="Обработка...")
            col_vid, col_map = st.columns([3, 2])
            frame_display = col_vid.empty()
            map_display   = col_map.empty() if slam_enabled else None

            tracker = (SimpleTracker(iou_threshold=iou_track_thresh)
                       if tracking_enabled else None)
            slam    = (SemanticSLAM(min_observations=slam_min_obs,
                                    min_features=slam_min_feat)
                       if slam_enabled else None)

            frame_idx       = 0
            processed_count = 0
            start_time      = time.time()
            svc_status      = "ОЖИДАНИЕ: Установите колодки"
            svc_color       = (0, 0, 255)
            _SLAM_INTERVAL  = 30  # update 3-D map every N processed frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % process_every_n == 0:
                    annotated, svc_status, svc_color = process_frame(
                        frame,
                        loaded.get("combo"), loaded.get("person"), loaded.get("pose"),
                        use_wheels, use_person, use_pose,
                        conf_wheels, conf_person, conf_pose_kpt,
                        combo_imgsz, line_thickness, font_scale,
                        svc_status, svc_color, show_status_bar,
                        tracker, show_track_ids, show_filt_stats,
                        filter_cfg=filter_cfg,
                        tracking_enabled=tracking_enabled,
                        slam=slam,
                        show_slam_stats=show_slam_overlay,
                    )
                    writer.write(annotated)
                    processed_count += 1

                    if processed_count % 15 == 0:
                        frame_display.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)

                    if (slam is not None and map_display is not None
                            and processed_count % _SLAM_INTERVAL == 0
                            and len(slam.point_cloud) > 0):
                        map_display.plotly_chart(
                            build_slam_plotly(slam, wheel_class_id, chock_class_id),
                            use_container_width=True,
                            key=f"slam_live_{processed_count}")

                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                    progress  = (frame_idx + 1) / total_frames
                    elapsed   = time.time() - start_time
                    remaining = (max(0, (elapsed / (frame_idx + 1) * total_frames) - elapsed)
                                 if frame_idx > 0 else 0)
                    progress_bar.progress(
                        min(progress, 1.0),
                        text=f"Кадр {frame_idx+1}/{total_frames} | "
                             f"Обработано: {processed_count} | ~{remaining:.0f} сек")
                frame_idx += 1

            cap.release()
            writer.release()
            total_time = time.time() - start_time
            progress_bar.progress(1.0, text="✅ Готово!")

            st.markdown("### 📊 Результаты")
            r1, r2, r3 = st.columns(3)
            r1.metric("⏱️ Время",   f"{total_time:.1f} сек")
            r2.metric("🎞️ Кадров",  f"{processed_count}")
            r3.metric("⚡ FPS",     (f"{processed_count/total_time:.1f}"
                                     if total_time > 0 else "—"))

            # ── SLAM final map ────────────────────────────────────────
            if slam is not None:
                total_obj, verified_obj = slam.get_stats()
                st.markdown("### 🗺️ Semantic SLAM — итог")
                sm1, sm2 = st.columns(2)
                sm1.metric("Всего объектов (треков)", total_obj)
                sm2.metric("Верифицированных",        verified_obj)

                final_fig = build_slam_plotly(slam, wheel_class_id, chock_class_id)
                if map_display is not None:
                    map_display.plotly_chart(final_fig, use_container_width=True,
                                            key="slam_final")
                else:
                    st.plotly_chart(final_fig, use_container_width=True,
                                   key="slam_final_standalone")

                html_buf = io.StringIO()
                final_fig.write_html(html_buf, include_plotlyjs="cdn")
                st.download_button(
                    "⬇️ Скачать 3D карту (HTML)",
                    html_buf.getvalue(),
                    "slam_3d_map.html", "text/html")

            # ── Output video ──────────────────────────────────────────
            st.markdown("### 🎬 Результат")
            h264_path = output_path.replace(".mp4", "_h264.mp4")
            os.system(f'ffmpeg -y -i "{output_path}" -vcodec libx264 '
                      f'-acodec aac "{h264_path}" -loglevel quiet')
            final = (h264_path
                     if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0
                     else output_path)

            st.video(final)
            with open(final, "rb") as f:
                st.download_button("⬇️ Скачать видео", f,
                                   "processed_video.mp4", "video/mp4",
                                   use_container_width=True)

            for p in [input_path, output_path, h264_path]:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except OSError:
                    pass
else:
    st.markdown("""
    <div style="border:2px dashed #4a5568;border-radius:16px;padding:3rem;
                text-align:center;background:linear-gradient(145deg,#f7fafc,#edf2f7);margin:2rem 0;">
        <p style="font-size:3rem;margin:0;">🎥</p>
        <p style="font-size:1.2rem;color:#4a5568;font-weight:600;">Загрузите видеофайл</p>
        <p style="font-size:0.9rem;color:#718096;">MP4, AVI, MOV, MKV, WMV</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#9ca3af;font-size:0.85rem;">'
    '✈️ Aircraft Detection v4 | YOLOv8 + MPII Pose + Tracking + Semantic SLAM 3D'
    '</div>',
    unsafe_allow_html=True,
)
