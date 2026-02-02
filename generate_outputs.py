"""
Unified Output Generator for Camera Orientation Project.
Generates both Video (MP4) and Interactive HTML exports from sensor data.
Both outputs are saved to the same timestamped results folder.
"""
print("DEBUG: generate_outputs starting...", flush=True)

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from scipy.spatial.transform import Rotation as R, Slerp
import os
import tqdm
import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data_loader import DataLoader
from core.noise_db import noise_db
from solvers.pytorch_solver import PyTorchSolver
from solvers.google_solver import GoogleEKF
from solvers.base_solver import OrientationTrajectory

# ==============================================================================
# Shared Phone Mesh Functions
# ==============================================================================

def get_phone_mesh_data_mpl(q, color_body='#333333', color_screen='#00AAFF', color_bump='#FF0000'):
    """
    Returns vertices/faces for Phone elements rotated by q (for Matplotlib 3D).
    q: [w, x, y, z]
    """
    width, height, depth = 1.0, 2.0, 0.2
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    
    def transform_box(box_x, box_y, box_z):
        verts = np.stack([box_x, box_y, box_z], axis=1)
        return r.apply(verts)

    # Body
    bx = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * (width / 2)
    by = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * (height / 2)
    bz = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * (depth / 2)
    body_verts = transform_box(bx, by, bz)
    
    # Screen (Offset in Z)
    screen_idx = [0, 1, 2, 3]
    normal_body = np.array([0, 0, 1])
    normal_world = r.apply(normal_body)
    offset = normal_world * 0.02
    s_verts = np.stack([bx[screen_idx], by[screen_idx], bz[screen_idx]], axis=1)
    screen_verts = r.apply(s_verts) + offset
    
    # Camera Bump (Red) - Protrude
    cx = np.array([-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3]) 
    cy = np.array([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9])
    cz = np.array([-0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2])
    bump_verts = transform_box(cx, cy, cz)
    
    return [
        (body_verts, [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]], color_body, 0.7),
        (screen_verts, [[0,1,2,3]], color_screen, 0.8),
        (bump_verts, [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]], color_bump, 1.0)
    ]


def get_phone_mesh_traces_plotly(q, name_suffix=""):
    """
    Returns plotly traces for phone at orientation q (w,x,y,z).
    """
    width, height, depth = 1.0, 2.0, 0.2
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    
    def transform(bx, by, bz):
        verts = np.stack([bx, by, bz], axis=1)
        return r.apply(verts)

    # Body
    bx = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * (width / 2)
    by = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * (height / 2)
    bz = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * (depth / 2)
    body_v = transform(bx, by, bz)
    
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
    trace_body = go.Mesh3d(
        x=body_v[:,0], y=body_v[:,1], z=body_v[:,2],
        i=i, j=j, k=k,
        color='#333333', name=f'Phone Body {name_suffix}', showscale=False,
        opacity=0.5,
        lighting=dict(ambient=0.5, diffuse=0.5)
    )
    
    # Camera Bump (Red)
    cx = np.array([-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3]) 
    cy = np.array([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9])
    cz = np.array([-0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2])
    bump_v = transform(cx, cy, cz)
    
    trace_bump = go.Mesh3d(
        x=bump_v[:,0], y=bump_v[:,1], z=bump_v[:,2],
        i=i, j=j, k=k,
        color='#FF0000', name=f'Cam Bump {name_suffix}', showscale=False,
        opacity=1.0,
        lighting=dict(ambient=0.9, diffuse=0.1)
    )
    
    # Screen (Blue)
    sx = bx[0:4] * 0.9
    sy = by[0:4] * 0.95
    sz = bz[0:4] + 0.01
    si = [0, 0]
    sj = [1, 2]
    sk = [2, 3]
    screen_v = transform(sx, sy, sz)
    
    trace_screen = go.Mesh3d(
        x=screen_v[:,0], y=screen_v[:,1], z=screen_v[:,2],
        i=si, j=sj, k=sk,
        color='#00AAFF', name=f'Screen {name_suffix}', showscale=False,
        lighting=dict(ambient=0.8, diffuse=0.5) 
    )
    
    return [trace_body, trace_bump, trace_screen]


def render_frame_mpl(ax, q, q_gt=None):
    """Render a single 3D frame for Matplotlib video."""
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_aspect('equal')
    
    # Render GT Ghost (if exists)
    if q_gt is not None:
        meshes_gt = get_phone_mesh_data_mpl(q_gt, color_body='#AAAAAA', color_screen='#CCCCFF', color_bump='#FFAAAA')
        for verts, faces, color, alpha in meshes_gt:
            polys = [[verts[i] for i in face] for face in faces]
            coll = Poly3DCollection(polys, facecolors=color, edgecolors='gray', linewidths=0.5, alpha=alpha * 0.3)
            ax.add_collection3d(coll)

    # Render Estimate
    meshes = get_phone_mesh_data_mpl(q)
    for verts, faces, color, alpha in meshes:
        polys = [[verts[i] for i in face] for face in faces]
        coll = Poly3DCollection(polys, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
        ax.add_collection3d(coll)
        
    # Add axes arrows to show World Frame
    ax.quiver(0,0,0, 1.5,0,0, color='r', linewidth=2.0)
    ax.quiver(0,0,0, 0,1.5,0, color='g', linewidth=2.0)
    ax.quiver(0,0,0, 0,0,1.5, color='b', linewidth=2.0)
    
    # Text Labels
    ax.text(1.6, 0, 0, "East", color='red', fontsize=12, fontweight='bold')
    ax.text(0, 1.6, 0, "North", color='green', fontsize=12, fontweight='bold')
    ax.text(0, 0, 1.6, "Up", color='blue', fontsize=12, fontweight='bold')
    
    # Legend
    if q_gt is not None:
        ax.text2D(0.05, 0.95, "Solid: Estimate\nTransp: Ground Truth", transform=ax.transAxes, color='black', fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


# ==============================================================================
# Video Generation
# ==============================================================================

def generate_video(data, trajectory, video_path, output_path, limit_seconds=None):
    """Generate MP4 video output."""
    print("Generating Video...")
    
    # Prepare Interpolators
    rots = R.from_quat(trajectory.quaternions[:, [1,2,3,0]])
    key_times = data.time
    slerp = Slerp(key_times, rots)
    
    slerp_gt = None
    if getattr(data, 'orientation', None) is not None and len(data.orientation) > 0:
        rots_gt = R.from_quat(data.orientation[:, [1,2,3,0]])
        slerp_gt = Slerp(key_times, rots_gt)
    
    # Video Processing
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Matplotlib Figure
    fig = plt.figure(figsize=(12, 8), dpi=100) 
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 2])
    
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_r = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])
    ax_y = fig.add_subplot(gs[2, 1])
    
    # Pre-calculate RPY
    rpy = rots.as_euler('xyz', degrees=True)
    timestamps = data.time
    
    # GT RPY
    rpy_gt = None
    if getattr(data, 'orientation', None) is not None and len(data.orientation) > 0:
        rpy_gt = R.from_quat(data.orientation[:, [1,2,3,0]]).as_euler('xyz', degrees=True)

    line_cursors = []
    
    # Uncertainty (2-sigma)
    if trajectory.covariances is not None:
        sigmas = np.sqrt(np.diagonal(trajectory.covariances, axis1=1, axis2=2))
        two_sigma = 2.0 * sigmas
    else:
        two_sigma = None

    for ax_2d, idx, label, col in zip([ax_r, ax_p, ax_y], [0, 1, 2], ['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b']):
        ax_2d.set_title(label, fontsize=10, pad=2)
        
        # Uncertainty Band (clamped to min 1.0 deg)
        if two_sigma is not None:
            sigma_vis = np.maximum(two_sigma[:, idx], 1.0)
            ax_2d.fill_between(timestamps, rpy[:, idx] - sigma_vis, rpy[:, idx] + sigma_vis, 
                               color=col, alpha=0.2, label='2$\\sigma$')
        
        ax_2d.plot(timestamps, rpy[:, idx], color=col, alpha=0.8, label='Est')
        if rpy_gt is not None:
            ax_2d.plot(timestamps, rpy_gt[:, idx], 'k--', alpha=0.5, label='GT')
        ax_2d.grid(True)
        ax_2d.legend(loc='upper right', fontsize='x-small')
        
        line = ax_2d.axvline(x=0, color='k', linewidth=1.5)
        line_cursors.append(line)

    canvas = FigureCanvas(fig)
    canvas.draw()
    dummy_buf = np.asarray(canvas.buffer_rgba())
    plot_h, plot_w, _ = dummy_buf.shape
    
    scale = height / plot_h
    final_plot_w = int(plot_w * scale)
    out_width = width + final_plot_w
    out_height = height
    
    print(f"Video Output Size: {out_width}x{out_height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    limit_frames = int(limit_seconds * fps) if limit_seconds else total_frames
    print(f"Processing {limit_frames} frames...")
    
    for i in tqdm.tqdm(range(limit_frames), unit="frame", desc="Video"):
        ret, frame = cap.read()
        if not ret: break
        
        t = i / fps
        t_clamp = np.clip(t, key_times[0], key_times[-1])
        
        try:
            r_interp = slerp(t_clamp)
            q_interp = r_interp.as_quat()
            q_viz = [q_interp[3], q_interp[0], q_interp[1], q_interp[2]]
        except Exception:
            q_viz = [1, 0, 0, 0]

        q_gt_viz = None
        if slerp_gt is not None:
            try:
                r_gt_interp = slerp_gt(t_clamp)
                q_gt = r_gt_interp.as_quat()
                q_gt_viz = [q_gt[3], q_gt[0], q_gt[1], q_gt[2]]
            except:
                pass

        render_frame_mpl(ax_3d, q_viz, q_gt_viz)
        
        for line in line_cursors:
            line.set_xdata([t_clamp, t_clamp])
        
        canvas.draw()
        try:
            buf = np.asarray(canvas.buffer_rgba())
            img_plot = buf[:, :, :3]
            img_plot = cv2.resize(img_plot, (final_plot_w, height))
            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
            combined = np.hstack((frame, img_plot))
            out.write(combined)
        except Exception as e:
            print(f"Frame error: {e}") 
        
    cap.release()
    out.release()
    plt.close(fig)
    print(f"Video saved to: {output_path}")


# ==============================================================================
# HTML Generation
# ==============================================================================

def generate_html(data, trajectory, output_path, step=5, limit_seconds=None):
    """Generate interactive Plotly HTML output."""
    print("Generating HTML Viewer...")
    
    timestamps = data.time
    quats = trajectory.quaternions
    
    # GT Data
    rpy_gt = None
    if getattr(data, 'orientation', None) is not None and len(data.orientation) > 0:
        q_gt = data.orientation
        if q_gt.shape[1] == 4:
            rots_gt = R.from_quat(q_gt[:, [1,2,3,0]])
            rpy_gt = rots_gt.as_euler('xyz', degrees=True)
    
    # Calculate RPY
    rots = R.from_quat(quats[:, [1,2,3,0]])
    rpy = rots.as_euler('xyz', degrees=True)
    
    N = len(timestamps)
    if limit_seconds:
        limit_idx = int(limit_seconds * 30.0)
        if limit_idx < N:
            timestamps = timestamps[:limit_idx]
            quats = quats[:limit_idx]
            rpy = rpy[:limit_idx]
            if rpy_gt is not None:
                rpy_gt = rpy_gt[:limit_idx]
            if trajectory.covariances is not None:
                trajectory.covariances = trajectory.covariances[:limit_idx]
            N = limit_idx
            
    # Uncertainty
    two_sigma = None
    if trajectory.covariances is not None:
        sigmas = np.sqrt(np.diagonal(trajectory.covariances, axis1=1, axis2=2))
        two_sigma = 2.0 * sigmas
            
    # Subsample for Frames
    indices = np.arange(0, N, step)
    
    # Create Figure
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "xy"}],
            [None,                            {"type": "xy"}],
            [None,                            {"type": "xy"}]
        ],
        column_widths=[0.6, 0.4],
        subplot_titles=("3D View", "Roll", "Pitch", "Yaw")
    )
    
    # Initial Traces
    q0 = quats[0]
    
    # GT Ghost (Transparent)
    if rpy_gt is not None:
        q0_gt = data.orientation[0]
        t_gt = get_phone_mesh_traces_plotly(q0_gt, name_suffix=" (GT)")
        for t in t_gt:
            t.color = 'lightgrey'
            t.opacity = 0.1
            t.lighting = dict(ambient=0.9)
        for t in t_gt:
            fig.add_trace(t, row=1, col=1)
        
    traces_3d = get_phone_mesh_traces_plotly(q0)
    for t in traces_3d:
        fig.add_trace(t, row=1, col=1)
        
    # Axes
    fig.add_trace(go.Scatter3d(x=[0,1.5], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=5), name='East'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,1.5], z=[0,0], mode='lines', line=dict(color='green', width=5), name='North'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,1.5], mode='lines', line=dict(color='blue', width=5), name='Up'), row=1, col=1)

    # RPY Traces
    colors = ['red', 'green', 'blue']
    names = ['Roll', 'Pitch', 'Yaw']
    
    for i in range(3):
        row = i + 1
        col = 2
        color = colors[i]
        
        # Uncertainty (clamped to min 1.0 deg)
        if two_sigma is not None:
            rgba = 'rgba(255,0,0,0.2)' if i==0 else 'rgba(0,255,0,0.2)' if i==1 else 'rgba(0,0,255,0.2)'
            sigma_vis = np.maximum(two_sigma[:, i], 1.0)
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=rpy[:,i] - sigma_vis,
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
                legendgroup=f'unc_{names[i]}'
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=rpy[:,i] + sigma_vis,
                fill='tonexty', fillcolor=rgba,
                mode='lines', line=dict(width=0), name=f'2σ', showlegend=(i==0),
                legendgroup=f'unc_{names[i]}'
            ), row=row, col=col)

        # Mean
        fig.add_trace(go.Scatter(x=timestamps, y=rpy[:,i], mode='lines', line=dict(color=color), name=names[i]), row=row, col=col)
        
        # GT Line
        if rpy_gt is not None:
            fig.add_trace(go.Scatter(x=timestamps, y=rpy_gt[:,i], mode='lines', 
                                     line=dict(color='black', dash='dash', width=1), 
                                     name='GT' if i==0 else None, showlegend=(i==0)), row=row, col=col)
    
    # Time Markers
    fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[0]], y=[-180, 180], mode='lines', line=dict(color='black', width=2), name='Time'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[0]], y=[-180, 180], mode='lines', line=dict(color='black', width=2), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[0]], y=[-180, 180], mode='lines', line=dict(color='black', width=2), showlegend=False), row=3, col=2)
    
    # Frames
    frames = []
    print(f"Creating {len(indices)} HTML frames...")
    
    has_gt = (rpy_gt is not None)
    has_unc = (two_sigma is not None)
    
    for k in tqdm.tqdm(indices, desc="HTML"):
        t_curr = timestamps[k]
        q_curr = quats[k]
        
        current_data = []
        
        # GT Ghost Update
        if has_gt:
            q_gt_curr = data.orientation[k]
            gt_traces = get_phone_mesh_traces_plotly(q_gt_curr, name_suffix=" (GT)")
            for t in gt_traces:
                t.color = 'lightgrey'
                t.opacity = 0.1
            current_data.extend(gt_traces)
        
        mesh_traces = get_phone_mesh_traces_plotly(q_curr)
        current_data.extend(mesh_traces)
        
        # Time Markers
        line_x = [t_curr, t_curr]
        current_data.append(go.Scatter(x=line_x, y=[-180, 180]))
        current_data.append(go.Scatter(x=line_x, y=[-180, 180]))
        current_data.append(go.Scatter(x=line_x, y=[-180, 180]))
        
        frames.append(go.Frame(data=current_data, name=f"fr{k}"))
        
    # Calculate Trace Indices
    offset_mesh = 3 if has_gt else 0
    offset_axes = offset_mesh + 3
    offset_rpy = offset_axes + 3
    
    traces_per_subplot = 1
    if has_unc: traces_per_subplot += 2
    if has_gt: traces_per_subplot += 1
    
    offset_markers = offset_rpy + (3 * traces_per_subplot)
    
    frame_trace_indices = []
    if has_gt: frame_trace_indices.extend([0, 1, 2])
    frame_trace_indices.extend([offset_mesh, offset_mesh+1, offset_mesh+2])
    frame_trace_indices.extend([offset_markers, offset_markers+1, offset_markers+2])
    
    for f in frames:
        f.traces = frame_trace_indices

    # Layout
    fig.update_layout(
        title="Interactive Orientation Viewer",
        scene=dict(
            xaxis=dict(visible=False, range=[-1.5,1.5]),
            yaxis=dict(visible=False, range=[-1.5,1.5]),
            zaxis=dict(visible=False, range=[-1.5,1.5]),
            aspectmode='cube'
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, dict(frame=dict(duration=1000/30*step, redraw=True), fromcurrent=True)]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]
        )]
    )
    
    # Slider
    sliders = [dict(
        steps=[dict(method='animate',
                    args=[[f'fr{k}'], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                    label=f'{timestamps[k]:.1f}') for k in indices],
        active=0,
        transition=dict(duration=0),
        x=0, y=0, currentvalue=dict(font=dict(size=12), prefix='Time: ', visible=True, xanchor='right'),
        len=1.0
    )]
    fig.update_layout(sliders=sliders)
    
    fig.frames = frames
    
    fig.write_html(output_path, auto_play=False)
    print(f"HTML saved to: {output_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Video and HTML exports from sensor data.")
    parser.add_argument("--log", type=str, required=True, help="Path to sensor log file")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--limit", type=float, default=None, help="Limit output to N seconds")
    parser.add_argument("--device", type=str, default="pixel_10", help="Device ID for noise params")
    parser.add_argument("--offset", type=float, default=0.0, help="Sync offset in seconds")
    parser.add_argument("--outdir", type=str, default="results", help="Base output directory")
    parser.add_argument("--step", type=int, default=5, help="Subsample step for HTML frames")
    parser.add_argument("--skip-video", action="store_true", help="Skip video generation")
    parser.add_argument("--skip-html", action="store_true", help="Skip HTML generation")
    parser.add_argument("--solver", type=str, default="google", choices=["google", "pytorch"], help="Solver to use")
    args = parser.parse_args()
    
    # 0. Setup Output Directory
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.outdir, timestamp_str)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")
    
    video_output_path = os.path.join(result_dir, f"video_{timestamp_str}.mp4")
    html_output_path = os.path.join(result_dir, f"viewer_{timestamp_str}.html")
    
    # 1. Load Data
    print("Loading sensor data...")
    loader = DataLoader(target_freq=30.0)
    data = loader.load_data(args.log, args.video, additional_offset_s=args.offset)
    
    if len(data.time) == 0:
        print("Error: No data found. Check log path or sync logic.")
        return

    # 2. Solve
    print(f"Solving trajectory using {args.solver.upper()} solver...")
    noise_params = noise_db.get_params(args.device, is_indoor=True)
    
    if args.solver == "google":
        # Tuned parameters from validaton
        solver = GoogleEKF(gyro_bias_var=1e-6) 
        q_est, b_est = solver.solve(data.time, data.gyro, data.accel, data.mag)
        
        # Wrap in OrientationTrajectory
        # Approximating covariance as 0 for visualization if needed, or identity * small
        covs = np.zeros((len(q_est), 3, 3))
        # Fill diagonal with something visible? GoogleEKF tracks P but solve returns only state.
        # We can update solve to return P if we want uncertainty viz. 
        # For now, just return state.
        trajectory = OrientationTrajectory(
            timestamps=data.time,
            quaternions=q_est,
            covariances=None # Disable uncertainty viz for Google for now
        )
        
    else:
        solver = PyTorchSolver()
        trajectory = solver.solve(data, noise_params)
    
    # 3. Generate Outputs (HTML first - faster)
    if not args.skip_html:
        generate_html(data, trajectory, html_output_path, step=args.step, limit_seconds=args.limit)
    
    if not args.skip_video:
        generate_video(data, trajectory, args.video, video_output_path, limit_seconds=args.limit)
    
    print(f"\n=== All done! ===")
    print(f"Results in: {result_dir}")
    if not args.skip_video:
        print(f"  Video: {video_output_path}")
    if not args.skip_html:
        print(f"  HTML:  {html_output_path}")


if __name__ == "__main__":
    main()
