print("DEBUG: make_video starting...", flush=True)
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import argparse
from scipy.spatial.transform import Rotation as R
import os
import tqdm
import datetime
import shutil

from data_loader import DataLoader
from noise_db import noise_db
from solvers.pytorch_solver import PyTorchSolver


def get_phone_mesh_data(q, color_body='#333333', color_screen='#00AAFF', color_bump='#FF0000', alpha=0.9):
    """
    Returns vertices/faces for Phone elements rotated by q.
    """
    # Dimensions
    width, height, depth = 1.0, 2.0, 0.2
    
    # helper to rotate
    # q is [w, x, y, z] -> scalar first
    # scipy expects [x, y, z, w]
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    
    def transform_box(box_x, box_y, box_z):
        verts = np.stack([box_x, box_y, box_z], axis=1)
        rotated = r.apply(verts)
        return rotated

    # 1. Body
    bx = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * (width / 2)
    by = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * (height / 2)
    bz = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * (depth / 2)
    body_verts = transform_box(bx, by, bz)
    
    # 2. Screen (Offset in Z)
    screen_idx = [0, 1, 2, 3] # Front face
    normal_body = np.array([0, 0, 1])
    normal_world = r.apply(normal_body)
    offset = normal_world * 0.02
    
    s_verts = np.stack([bx[screen_idx], by[screen_idx], bz[screen_idx]], axis=1)
    screen_verts = r.apply(s_verts) + offset
    
    # 3. Camera Bump (Red) - PROTRUDE MORE
    cx = np.array([-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3]) 
    cy = np.array([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9])
    # Stick out from back (-0.1) to -0.2
    cz = np.array([-0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2])
    bump_verts = transform_box(cx, cy, cz)
    
    return [
        (body_verts, [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]], color_body, 0.7), # Low alpha body
        (screen_verts, [[0,1,2,3]], color_screen, 0.8),
        (bump_verts, [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]], color_bump, 1.0) # High alpha bump
    ]

def render_frame(ax, q, q_gt=None):
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_aspect('equal')
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Render GT Ghost (if exists)
    if q_gt is not None:
         # Use light grey/transparent style
         meshes_gt = get_phone_mesh_data(q_gt, color_body='#AAAAAA', color_screen='#CCCCFF', color_bump='#FFAAAA', alpha=0.3)
         for verts, faces, color, alpha in meshes_gt:
            polys = [[verts[i] for i in face] for face in faces]
            # Ghost: thinner lines, low alpha
            coll = Poly3DCollection(polys, facecolors=color, edgecolors='gray', linewidths=0.5, alpha=alpha)
            ax.add_collection3d(coll)

    # Render Estimate
    meshes = get_phone_mesh_data(q)
    for verts, faces, color, alpha in meshes:
        polys = [[verts[i] for i in face] for face in faces]
        coll = Poly3DCollection(polys, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
        ax.add_collection3d(coll)
        
    # Add axes arrows to show World Frame
    # Length 1.5, thicker arrowheads
    ax.quiver(0,0,0, 1.5,0,0, color='r', linewidth=2.0) # X
    ax.quiver(0,0,0, 0,1.5,0, color='g', linewidth=2.0) # Y
    ax.quiver(0,0,0, 0,0,1.5, color='b', linewidth=2.0) # Z (Up)
    
    # Text Labels
    ax.text(1.6, 0, 0, "East", color='red', fontsize=12, fontweight='bold')
    ax.text(0, 1.6, 0, "North", color='green', fontsize=12, fontweight='bold')
    ax.text(0, 0, 1.6, "Up", color='blue', fontsize=12, fontweight='bold')
    
    # Legend
    if q_gt is not None:
        ax.text2D(0.05, 0.95, "Solid: Estimate\nTransp: Ground Truth", transform=ax.transAxes, color='black', fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--limit", type=float, default=None, help="Limit video to N seconds")
    parser.add_argument("--device", type=str, default="pixel_10", help="Device ID for noise params")
    parser.add_argument("--offset", type=float, default=0.0, help="Sync offset in seconds")
    parser.add_argument("--outdir", type=str, default="results", help="Base output directory")
    args = parser.parse_args()
    
    # 0. Setup Output Directory
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.outdir, timestamp_str)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")
    
    video_filename = f"video_{timestamp_str}.mp4"
    output_path = os.path.join(result_dir, video_filename)
    
    # 1. Load Data
    print("Loading valid sensor data...")
    loader = DataLoader(target_freq=30.0) # Sync freq to ~30fps for smoothness
    data = loader.load_data(args.log, args.video, additional_offset_s=args.offset)
    
    if len(data.time) == 0:
        print("Error: No data found. Check log path or sync logic.")
        return

    # 2. Solve
    print("Solving trajectory...")
    noise_params = noise_db.get_params(args.device, is_indoor=True)
    solver = PyTorchSolver()
    trajectory = solver.solve(data, noise_params)
    
    # Interpolation Function
    # data.time is relative to video start.
    # We will iterate video frames and interp trajectory.
    from scipy.spatial.transform import Slerp
    
    # Create Slerp object
    rots = R.from_quat(trajectory.quaternions[:, [1,2,3,0]]) # x,y,z,w for Scipy
    key_times = data.time
    slerp = Slerp(key_times, rots)
    
    # Interpolator for GT (if available)
    slerp_gt = None
    if getattr(data, 'orientation', None) is not None and len(data.orientation) > 0:
         rots_gt = R.from_quat(data.orientation[:, [1,2,3,0]])
         slerp_gt = Slerp(data.unix_timestamps, rots_gt) # Use unix timestamps for absolute sync if needed?
         # Wait, data.time is synced relative time. Use data.time for simple correspondence.
         # Re-check data_loader.py: data.orientation is aligned with data.time (resampled).
         slerp_gt = Slerp(key_times, rots_gt)
    
    # 3. Video Processing
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup Output
    # Side-by-side dimension
    # Target height for plot = height of video
    # Target width for plot = height (square) => Total width = width + height
    # Setup Matplotlib Figure
    # 1.5 aspect ratio (12x8)
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
    
    # Prepare Uncertainty (2-sigma)
    # covariances is T x 3 x 3
    # Extract diagonal: Roll_var, Pitch_var, Yaw_var
    # Sigma = sqrt(var)
    if trajectory.covariances is not None:
         # Assuming trajectory is aligned with data.time? Yes, solver output.
         sigmas = np.sqrt(np.diagonal(trajectory.covariances, axis1=1, axis2=2)) # T x 3
         two_sigma = 2.0 * sigmas
    else:
         two_sigma = None

    for ax_2d, idx, label, col in zip([ax_r, ax_p, ax_y], [0, 1, 2], ['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b']):
        ax_2d.set_title(label, fontsize=10, pad=2)
        
        # Uncertainty Band
        if two_sigma is not None:
             # Clamp to minimum visibility width (1.0 degree)
             # User requested: "keep a minimum width so that it's always visible"
             sigma_vis = np.maximum(two_sigma[:, idx], 1.0)
             
             ax_2d.fill_between(timestamps, rpy[:, idx] - sigma_vis, rpy[:, idx] + sigma_vis, 
                                color=col, alpha=0.2, label='2$\sigma$')
        
        ax_2d.plot(timestamps, rpy[:, idx], color=col, alpha=0.8, label='Est')
        if rpy_gt is not None:
             ax_2d.plot(timestamps, rpy_gt[:, idx], 'k--', alpha=0.5, label='GT')
        ax_2d.grid(True)
        ax_2d.legend(loc='upper right', fontsize='x-small')
        
        # Cursor
        line = ax_2d.axvline(x=0, color='k', linewidth=1.5)
        line_cursors.append(line)

    canvas = FigureCanvas(fig)
    
    # Check plot size to init VideoWriter correctly
    canvas.draw()
    dummy_buf = np.asarray(canvas.buffer_rgba())
    plot_h, plot_w, _ = dummy_buf.shape
    
    # We will resize plot to match Video Height
    scale = height / plot_h
    final_plot_w = int(plot_w * scale)
    out_width = width + final_plot_w
    out_height = height
    
    print(f"Video Output Size: {out_width}x{out_height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    print("Rendering video...", flush=True)
    # for i in range(total_frames):
    limit_frames = int(args.limit * fps) if args.limit else total_frames
    print(f"Processing {limit_frames} frames...")
    
    for i in tqdm.tqdm(range(limit_frames), unit="frame"):
        # if i % 30 == 0:
        #     print(f"Processing frame {i}/{limit_frames}...", flush=True)
        ret, frame = cap.read()
        if not ret: break
        
        t = i / fps # Current video time
        
        # Interpolate Orientation
        # Clamp to range
        t_clamp = np.clip(t, key_times[0], key_times[-1])
        try:
            r_interp = slerp(t_clamp)
            q_interp = r_interp.as_quat() # x, y, z, w
            # Convert to w, x, y, z for our renderer
            q_viz = [q_interp[3], q_interp[0], q_interp[1], q_interp[2]]
        except Exception:
            q_viz = [1, 0, 0, 0]
            
        # Check if t is within Valid Data Range (Video might be longer than Log)
        if t < key_times[0] or t > key_times[-1]:
             # Fade out or indicate no data?
             pass 

        # Interpolate GT
        q_gt_viz = None
        if slerp_gt is not None:
             try:
                 r_gt_interp = slerp_gt(t_clamp)
                 q_gt = r_gt_interp.as_quat()
                 q_gt_viz = [q_gt[3], q_gt[0], q_gt[1], q_gt[2]]
             except:
                 pass

        # Render 3D Plot
        render_frame(ax_3d, q_viz, q_gt_viz)
        
        # Update 2D cursors
        for line in line_cursors:
            line.set_xdata([t_clamp, t_clamp])
        
        # Convert Plot to Image
        # Convert Plot to Image
        canvas.draw()
        try:
            buf = np.asarray(canvas.buffer_rgba())
            img_plot = buf[:, :, :3]
            
            # Resize
            img_plot = cv2.resize(img_plot, (final_plot_w, height))
            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
            
            combined = np.hstack((frame, img_plot))
            out.write(combined)
        except Exception as e:
            print(f"Frame error: {e}") 
        
    cap.release()
    out.release()
    plt.close(fig)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
