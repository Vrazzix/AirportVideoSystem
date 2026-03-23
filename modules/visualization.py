"""
Interactive 3-D point-cloud visualization for Semantic SLAM.
Uses Plotly so the chart is fully interactive (rotate / zoom / hover)
inside the Streamlit web app.
"""

import numpy as np
import plotly.graph_objects as go


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def cls_info(cls_id: int, wheel_cls: int, chock_cls: int) -> tuple:
    """Returns (display_name, hex_color) for a detection class id."""
    if cls_id == wheel_cls:
        return "Wheel", "#3c8cdc"
    if cls_id == chock_cls:
        return "Chock", "#ffa532"
    return "Person", "#32cd32"


def pseudo_xz(cx_mean: float, cy_mean: float, fw: int, fh: int) -> tuple:
    """
    Rough ground-plane (X, Z) estimate from image-space object centre.
    Used as a fallback when 3-D triangulation has not yet produced a pos3d.
    """
    z = max((1.0 - cy_mean / fh) * 10.0, 0.05)
    x = (cx_mean / fw - 0.5) * z * (fw / fh) * 2.0
    return float(x), float(z)


# ──────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ──────────────────────────────────────────────────────────────────────────────

def build_slam_plotly(slam, wheel_cls: int, chock_cls: int) -> go.Figure:
    """
    Build an interactive Plotly 3-D scatter figure from a SemanticSLAM instance.

    Layers:
      1. Sparse point cloud (triangulated SIFT features) — environment structure.
      2. Camera trajectory — 3-D polyline with current-position marker.
      3. Verified objects — large coloured spheres with hover info.
      4. Unverified objects — small hollow grey markers.

    Args:
        slam:      A SemanticSLAM instance (must have .get_map_data()).
        wheel_cls: Class id for wheels.
        chock_cls: Class id for chocks.
    Returns:
        plotly.graph_objects.Figure
    """
    data = slam.get_map_data()
    objects  = data["objects"]
    cam_traj = data["cam_traj"]
    cloud    = data["point_cloud"]
    fw, fh   = data["frame_w"], data["frame_h"]

    traces = []

    # ── 1. Point cloud ────────────────────────────────────────────────
    if cloud:
        step = max(1, len(cloud) // 4000)   # subsample for Plotly performance
        sampled = cloud[::step]

        bg_pts = [(p[0], p[1], p[2]) for p in sampled if p[3] < 0]
        if bg_pts:
            bx, by, bz = zip(*bg_pts)
            traces.append(go.Scatter3d(
                x=bx, y=by, z=bz, mode="markers",
                marker=dict(size=1.5, color="#4a5568", opacity=0.4),
                name=f"Environment ({len(bg_pts)} pts)",
                hoverinfo="skip",
            ))

        cls_pts: dict = {}
        for p in sampled:
            if p[3] >= 0:
                cls_pts.setdefault(int(p[3]), []).append((p[0], p[1], p[2]))
        for cid, pts in cls_pts.items():
            name, color = cls_info(cid, wheel_cls, chock_cls)
            ox, oy, oz = zip(*pts)
            traces.append(go.Scatter3d(
                x=ox, y=oy, z=oz, mode="markers",
                marker=dict(size=2.5, color=color, opacity=0.7),
                name=f"{name} pts ({len(pts)})",
                hoverinfo="skip",
            ))

    # ── 2. Camera trajectory ──────────────────────────────────────────
    traj_pts = [
        (float(p[0]), float(p[1]), float(p[2]))
        for p in cam_traj
        if isinstance(p, np.ndarray) and p.shape[0] >= 3
    ]
    if traj_pts:
        tx, ty, tz = zip(*traj_pts)
        traces.append(go.Scatter3d(
            x=tx, y=ty, z=tz, mode="lines",
            line=dict(color="#ffffff", width=3),
            name="Camera path",
            hoverinfo="skip",
        ))
        traces.append(go.Scatter3d(
            x=[tx[-1]], y=[ty[-1]], z=[tz[-1]], mode="markers",
            marker=dict(size=8, color="#00ff88", symbol="diamond"),
            name="Camera (now)",
            hovertemplate="Camera<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
        ))

    # ── 3. Objects ────────────────────────────────────────────────────
    for obj in objects:
        tid     = obj["track_id"]
        name, color = cls_info(obj["cls"], wheel_cls, chock_cls)
        conf    = obj["conf_avg"]

        if obj["pos3d"] is not None:
            p = obj["pos3d"]
            ox, oy, oz = float(p[0]), float(p[1]), float(p[2])
        else:
            ox, oz = pseudo_xz(obj["cx_mean"], obj["cy_mean"], fw, fh)
            oy = 0.0

        if obj["verified"]:
            sz = max(10, min(20, 10 + obj["obs"] // 3))
            traces.append(go.Scatter3d(
                x=[ox], y=[oy], z=[oz], mode="markers+text",
                marker=dict(size=sz, color=color, opacity=0.95,
                            line=dict(width=2, color="#ffffff")),
                text=[f"{name}#{tid}"],
                textposition="top center",
                textfont=dict(size=10, color=color),
                name=f"{name}#{tid} ({conf:.0%})",
                hovertemplate=(
                    f"{name} #{tid}<br>conf: {conf:.1%}<br>obs: {obj['obs']}<br>"
                    "x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra>VERIFIED</extra>"
                ),
                showlegend=False,
            ))
        else:
            traces.append(go.Scatter3d(
                x=[ox], y=[oy], z=[oz], mode="markers",
                marker=dict(size=4, color="#666666", opacity=0.4, symbol="circle-open"),
                name=f"?#{tid}",
                hovertemplate=(
                    f"{name} #{tid} (unverified)<br>obs: {obj['obs']}<extra></extra>"
                ),
                showlegend=False,
            ))

    # ── Layout ────────────────────────────────────────────────────────
    n_ver = sum(1 for o in objects if o["verified"])
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="#0f172a",
                       gridcolor="#1e3a5f", showspikes=False),
            yaxis=dict(title="Y", backgroundcolor="#0f172a",
                       gridcolor="#1e3a5f", showspikes=False),
            zaxis=dict(title="Z (depth)", backgroundcolor="#0f172a",
                       gridcolor="#1e3a5f", showspikes=False),
            bgcolor="#0f172a",
            aspectmode="data",
        ),
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        title=dict(
            text=(f"Semantic SLAM 3D  |  {len(cloud)} pts  |  "
                  f"{len(traj_pts)} cam poses  |  {n_ver} verified"),
            font=dict(size=14),
        ),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=35, b=0),
        height=550,
    )
    return fig
