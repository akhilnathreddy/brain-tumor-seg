"""Patient-level views shared by the app and the results notebook."""
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import to_rgb
from skimage import measure

from src.utils import REGION_COLORS, REGIONS


def _rot(a):
    return np.rot90(a)  # array (x, y) -> conventional axial view


def gray(slice2d, brain2d=None):
    """Grayscale RGB uint8 of one slice, contrast from brain pixels, black background."""
    s = slice2d.astype(np.float32)
    m = brain2d if brain2d is not None else s != 0
    lo, hi = np.percentile(s[m], [1, 99]) if m.any() else (0.0, 1.0)
    g = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * m
    return (_rot(np.repeat(g[..., None], 3, -1)) * 255).astype(np.uint8)


def overlay(slice2d, regions2d, show=(True, True, True), alpha=0.45, brain2d=None):
    """Grayscale slice with WT/TC/ET filled in their colors (innermost region on top)."""
    rgb = gray(slice2d, brain2d).astype(np.float32)
    for i, name in enumerate(REGIONS):
        if show[i]:
            m = _rot(regions2d[i]).astype(bool)
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(to_rgb(REGION_COLORS[name])) * 255
    return rgb.astype(np.uint8)


def uncertainty_rgb(u2d, brain2d=None):
    """Uncertainty in [0,1] as a light-to-dark blue map; outside the brain stays black."""
    rgb = cm.Blues(np.clip(u2d.astype(np.float32), 0, 1))[..., :3]
    if brain2d is not None:
        rgb[~brain2d] = 0
    return (_rot(rgb) * 255).astype(np.uint8)


def area_chart(regions3d, z_current=None, height=260):
    """Tumor area per axial slice (cm², 1 mm voxels) for each region."""
    fig = go.Figure()
    zs = np.arange(regions3d.shape[-1])
    for i, name in enumerate(REGIONS):
        area = regions3d[i].sum((0, 1)) / 100.0
        fig.add_trace(go.Scatter(x=zs, y=area, name=name, mode="lines",
                                 line={"color": REGION_COLORS[name], "width": 2},
                                 hovertemplate=f"{name}: %{{y:.1f}} cm²<br>slice %{{x}}<extra></extra>"))
    if z_current is not None:
        fig.add_vline(x=z_current, line_width=1, line_dash="dot", line_color="#888780")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      xaxis_title="slice (z)", yaxis_title="area (cm²)",
                      legend=dict(orientation="h", y=1.12, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    return fig


def _mesh(mask, step, color, opacity, name):
    if mask.sum() < 20:
        return None
    verts, faces, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, step_size=step)
    verts -= 1
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color=color, opacity=opacity, name=name, showlegend=True, hoverinfo="name")


def view_3d(regions3d, brain3d=None, step=2, height=480):
    """Interactive 3D surfaces of the regions inside a faint brain outline."""
    traces = []
    if brain3d is not None:
        traces.append(_mesh(brain3d, 4, "#b4b2a9", 0.07, "brain"))
    for i, (name, op) in enumerate(zip(REGIONS, (0.25, 0.5, 0.9))):
        traces.append(_mesh(regions3d[i], step, REGION_COLORS[name], op, name))
    fig = go.Figure([t for t in traces if t is not None])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False,
                                 zaxis_visible=False),
                      legend=dict(orientation="h", y=0.02, x=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig
