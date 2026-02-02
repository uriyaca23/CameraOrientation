import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
import tqdm
import datetime
import os
from data_loader import DataLoader
from noise_db import noise_db
from solvers.pytorch_solver import PyTorchSolver

def get_phone_mesh_traces(q, name_suffix=""):
    """
    Returns plotly traces for phone at orientation q (w,x,y,z).
    """
    # Dimensions
    width, height, depth = 1.0, 2.0, 0.2
    
    # helper to rotate
    # q is [w, x, y, z] -> scalar first
    # scipy expects [x, y, z, w]
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    
    def transform(bx, by, bz):
        verts = np.stack([bx, by, bz], axis=1)
        return r.apply(verts)

    # 1. Body
    bx = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * (width / 2)
    by = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * (height / 2)
    bz = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * (depth / 2)
    body_v = transform(bx, by, bz)
    
    # Simple Mesh using Mesh3d is fastest for many frames? 
    # Or Scatter3d lines? Mesh3d is better for solid look.
    # Vertices for box
    # 8 corners. 12 triangles (2 per face * 6 faces).
    # i, j, k indices...
    # Indices for a cube
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
    trace_body = go.Mesh3d(
        x=body_v[:,0], y=body_v[:,1], z=body_v[:,2],
        i=i, j=j, k=k,
        color='#333333', name=f'Phone Body {name_suffix}', showscale=False,
        opacity=0.5, # Semi-transparent
        lighting=dict(ambient=0.5, diffuse=0.5)
    )
    
    # 2. Camera Bump (Red)
    cx = np.array([-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3]) 
    cy = np.array([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9])
    cz = np.array([-0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2]) # Protrude to -0.2
    bump_v = transform(cx, cy, cz)
    
    trace_bump = go.Mesh3d(
        x=bump_v[:,0], y=bump_v[:,1], z=bump_v[:,2],
        i=i, j=j, k=k,
        color='#FF0000', name=f'Cam Bump {name_suffix}', showscale=False,
        opacity=1.0,
        lighting=dict(ambient=0.9, diffuse=0.1) # Bright red
    )
    
    # 3. Screen (Blue) - Front Face (indices 0,1,2,3)
    # Offset slightly in Z (body frame)
    # bx/by/bz definitions:
    # 0: -x, -y, +z
    # 1: +x, -y, +z
    # 2: +x, +y, +z
    # 3: -x, +y, +z
    # This is the front face.
    
    sx = bx[0:4] * 0.9 # Slightly smaller
    sy = by[0:4] * 0.95
    sz = bz[0:4] + 0.01 # Slightly offset
    
    # Needs two triangles
    # 0,1,2 and 0,2,3
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

def add_uncertainty_traces(fig, timestamps, rpy, two_sigma, row_offset=0, col_offset=0):
    if two_sigma is None: return
    
    # Plot borders of the band
    # Upper
    for i, (name, color) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['red', 'green', 'blue'])):
         row = i + 1 + row_offset
         col = 1 + col_offset
         
         upper = rpy[:, i] + two_sigma[:, i]
         lower = rpy[:, i] - two_sigma[:, i]
         
         # Filled area in Plotly is done by plotting upper line, then lower line with fill='tonexty'
         # But here we have the mean line already.
         # So: Lower (transparent) -> Upper (fill to prev).
         
         fig.add_trace(go.Scatter(
             x=timestamps, y=lower,
             mode='lines', line=dict(width=0),
             showlegend=False, hoverinfo='skip'
         ), row=row, col=col)
         
         
         # Calculate RGBA string
         # Simple explicit map
         if color == 'red':
             rgba = 'rgba(255, 0, 0, 0.2)'
         elif color == 'green':
             rgba = 'rgba(0, 255, 0, 0.2)'
         elif color == 'blue':
             rgba = 'rgba(0, 0, 255, 0.2)'
         else:
             rgba = 'rgba(0, 0, 0, 0.2)'
             
         fig.add_trace(go.Scatter(
             x=timestamps, y=upper,
             fill='tonexty', fillcolor=rgba,
             mode='lines', line=dict(width=0),
             name=f'{name} 2σ', showlegend=True
         ), row=row, col=col)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="orientation_viewer.html", help="Output HTML filename")
    parser.add_argument("--outdir", type=str, default="results", help="Base output directory")
    parser.add_argument("--offset", type=float, default=0.0, help="Sync offset in seconds")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--step", type=int, default=5, help="Subsample step for HTML frames to reduce size")
    args = parser.parse_args()
    
    # Setup Output
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Or reuse explicit timestamp if we want to match video?
    # Actually, make_video makes a TIMESTAMP folder.
    # We should probably follow suit or use latest?
    # Let's create a new timestamp folder to prevent overwriting or potential race conditions if run in parallel (though different files).
    # Ideally, they should go to same folder.
    # But for now, separate is safer. 
    result_dir = os.path.join(args.outdir, timestamp_str)
    os.makedirs(result_dir, exist_ok=True)
    
    # If output filename has no path, join with result_dir
    if not os.path.dirname(args.output):
        output_path = os.path.join(result_dir, args.output)
    else:
        output_path = args.output
        
    print(f"Saving to {output_path}")

    # 1. Load & Solve
    print("Loading data...")
    loader = DataLoader(target_freq=30.0)
    data = loader.load_data(args.log, args.video, additional_offset_s=args.offset)
    
    print("Solving...")
    solver = PyTorchSolver()
    noise_params = noise_db.get_params("pixel_10", is_indoor=True)
    trajectory = solver.solve(data, noise_params)
    
    # 2. Prepare Plotly
    # Layout: 
    # Left: 3D Scene
    # Right: RPY Subplots (3 rows)
    print("Generating HTML...")
    
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
    
    timestamps = data.time
    quats = trajectory.quaternions # w,x,y,z
    
    # GT Data
    rpy_gt = None
    if getattr(data, 'orientation', None) is not None and len(data.orientation) > 0:
         q_gt = data.orientation # w,x,y,z ideally
         # Check shape
         if q_gt.shape[1] == 4:
             rots_gt = R.from_quat(q_gt[:, [1,2,3,0]])
             rpy_gt = rots_gt.as_euler('xyz', degrees=True)
         else:
             print("Warning: GT orientation shape invalid.")
    
    # Calculate RPY
    rots = R.from_quat(quats[:, [1,2,3,0]])
    rpy = rots.as_euler('xyz', degrees=True)
    
    # Limit
    N = len(timestamps)
    if args.limit:
        limit_idx = int(args.limit * 30.0) # approx
        if limit_idx < N:
            timestamps = timestamps[:limit_idx]
            quats = quats[:limit_idx]
            rpy = rpy[:limit_idx]
            N = limit_idx
            
            quats = quats[:limit_idx]
            rpy = rpy[:limit_idx]
            if rpy_gt is not None:
                rpy_gt = rpy_gt[:limit_idx]
            if trajectory.covariances is not None:
                trajectory.covariances = trajectory.covariances[:limit_idx] # Hacky but needed if we use it
            N = limit_idx
            
    # Uncertainty
    two_sigma = None
    if trajectory.covariances is not None:
         sigmas = np.sqrt(np.diagonal(trajectory.covariances, axis1=1, axis2=2))
         two_sigma = 2.0 * sigmas
            
    # Subsample for Frames
    indices = np.arange(0, N, args.step)
    
    # Initial Traces (t=0)
    q0 = quats[0]
    traces_3d = get_phone_mesh_traces(q0)
    
    # Initial GT Ghost (Transparent)
    traces_gt = []
    if rpy_gt is not None:
         # Need q_gt slice
         q0_gt = data.orientation[0]
         # Use a special Ghost function or just modify colors?
         # Reuse get_phone_mesh_traces but override colors?
         # Easier to duplicate/modify helper or post-process traces.
         # Let's clean up get_phone_mesh_traces to accept colors?
         # Or post-process:
         t_gt = get_phone_mesh_traces(q0_gt, name_suffix=" (GT)")
         # Modify colors to be transparent grey/ghostly
         for t in t_gt:
             t.color = 'lightgrey'
             t.opacity = 0.1
             t.lighting = dict(ambient=0.9)
         traces_gt = t_gt
         
    for t in traces_gt:
        fig.add_trace(t, row=1, col=1)
        
    for t in traces_3d:
        fig.add_trace(t, row=1, col=1)
        
    # Axes
    # Use explicit indices if possible? Plotly adds primarily.
    # Added: Body, Bump, Screen. (3 traces)
    fig.add_trace(go.Scatter3d(x=[0,1.5], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=5), name='East'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,1.5], z=[0,0], mode='lines', line=dict(color='green', width=5), name='North'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,1.5], mode='lines', line=dict(color='blue', width=5), name='Up'), row=1, col=1)

    # RPY Traces (Full Lines)
    # RPY Traces (Full Lines)
    # Add Uncertainty First (so lines are on top)
    # Actually Plotly order is by trace addition.
    
    # We need to add traces per subplot manually if using add_uncertainty_traces helper?
    # Or just loop here.
    
    colors = ['red', 'green', 'blue']
    names = ['Roll', 'Pitch', 'Yaw']
    
    for i in range(3):
        row = i + 1
        col = 2
        color = colors[i]
        
        # Uncertainty
        if two_sigma is not None:
            # Color map
            rgba = 'rgba(255,0,0,0.2)' if i==0 else 'rgba(0,255,0,0.2)' if i==1 else 'rgba(0,0,255,0.2)'
            
            # Clamp width (User request)
            sigma_vis = np.maximum(two_sigma[:, i], 1.0)
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=rpy[:,i] - sigma_vis,
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=rpy[:,i] + sigma_vis,
                fill='tonexty', fillcolor=rgba,
                mode='lines', line=dict(width=0), name=f'2σ', showlegend=(i==0)
            ), row=row, col=col)

        # Mean
        fig.add_trace(go.Scatter(x=timestamps, y=rpy[:,i], mode='lines', line=dict(color=color), name=names[i]), row=row, col=col)
        
        # GT Line
        if rpy_gt is not None:
             fig.add_trace(go.Scatter(x=timestamps, y=rpy_gt[:,i], mode='lines', 
                                      line=dict(color='black', dash='dash', width=1), 
                                      name='GT' if i==0 else None, showlegend=(i==0)), row=row, col=col)
    
    # Current Time Markers (Vertical Lines)
    # We use a Scatter point tracing the line? Or just a shape? 
    # Frames can update shapes? Yes.
    # Let's use a vertical line trace that moves.
    fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[0]], y=[-180, 180], mode='lines', line=dict(color='black', width=2), name='Time'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[0]], y=[-180, 180], mode='lines', line=dict(color='black', width=2), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[0]], y=[-180, 180], mode='lines', line=dict(color='black', width=2), showlegend=False), row=3, col=2)
    
    # Frames
    frames = []
    print(f"Creating {len(indices)} frames...")
    
    # Indices of traces in data list
    # 0,1: Phone Body, Bump
    # 2,3,4: Axes (Fixed)
    # 5,6,7: RPY Lines (Fixed)
    # 8,9,10: Time Markers (Update)
    
    for k in tqdm.tqdm(indices, desc="Generating HTML Frames"):
        t_curr = timestamps[k]
        q_curr = quats[k]
        
        mesh_traces = get_phone_mesh_traces(q_curr) # New 3D positions
        
        current_data = []
        # Add GT Ghost Update
        if rpy_gt is not None:
             q_gt_curr = data.orientation[k]
             gt_traces = get_phone_mesh_traces(q_gt_curr, name_suffix=" (GT)")
             for t in gt_traces:
                 t.color = 'lightgrey'
                 t.opacity = 0.1
             current_data.extend(gt_traces)
        
        current_data.extend(mesh_traces)
        
        # update vertical lines
        line_x = [t_curr, t_curr]
        
        current_data.append(go.Scatter(x=line_x, y=[-180, 180])) # Marker 1
        current_data.append(go.Scatter(x=line_x, y=[-180, 180])) # Marker 2
        current_data.append(go.Scatter(x=line_x, y=[-180, 180])) # Marker 3
        
        frames.append(go.Frame(
            data=current_data,
            traces=list(range(len(current_data))), # Re-calculate indices below?
            # actually we don't need explicit 'traces' if we structure it right or if we update ALL traces.
            # But we have static traces (Axes, Lines) that we DON'T want to overwrite.
            # We need to map `current_data` items to the `traces` indices in the figure.
            # Figure Traces Order:
            # 1. GT Ghost (3 traces) [Optional]
            # 2. Estimate Mesh (3 traces)
            # 3. Axes (3 traces)
            # 4. RPY Plots (Multiple)
            # 5. Time Markers (3 traces)
            
            # So if GT exists:
            # Indices [0, 1, 2] -> GT Ghost
            # Indices [3, 4, 5] -> Est Mesh
            # ...
            # Markers are at the END.
            
            # Let's count total traces first.
            name=f"fr{k}"
        ))
        
    # Calculate marker indices
    # Body(0), Bump(1), Screen(2), Axes(3,4,5) -> 6 traces
    # RPY start at 6.
    # If Unc: 3 subplots * 3 traces = 9. End at 15.
    # Markers start at 15.
    # If No Unc: 3 subplots * 1 trace = 3. End at 9.
    # Markers start at 9.
    
    # Calculate Loop Indices
    # Total Traces = 
    #   GT Ghost (3) [If GT]
    #   Est Mesh (3)
    #   Axes (3)
    #   RPY Subplots:
    #       Per subplot: Unc(2) + Mean(1) + GT(1) = 4 [If Unc & GT]
    #                    Unc(2) + Mean(1) = 3 [If Unc]
    #                    Mean(1) + GT(1) = 2 [If GT]
    #                    Mean(1) = 1
    #       Total RPY = 3 * per_subplot
    #   Markers (3)
    
    # We construct the frame 'traces' list based on this structure.
    # We update: GT Ghost, Est Mesh, Markers.
    # GT Ghost Indices: 0, 1, 2 (If GT)
    # Est Mesh Indices: 3, 4, 5 (If GT) else 0, 1, 2
    # Marker Indices: End - 3, End - 2, End - 1
    
    has_gt = (rpy_gt is not None)
    has_unc = (two_sigma is not None)
    
    offset_mesh = 3 if has_gt else 0
    offset_axes = offset_mesh + 3
    offset_rpy = offset_axes + 3
    
    traces_per_subplot = 1 # Mean
    if has_unc: traces_per_subplot += 2 # Unc fill + Line
    if has_gt: traces_per_subplot += 1 # GT Line
    
    offset_markers = offset_rpy + (3 * traces_per_subplot)
    
    # Frame Traces List
    # [GT 0,1,2, Est 0,1,2, Marker 0,1,2] -> Mapped to figure indices
    
    frame_trace_indices = []
    if has_gt: frame_trace_indices.extend([0, 1, 2])
    frame_trace_indices.extend([offset_mesh, offset_mesh+1, offset_mesh+2])
    frame_trace_indices.extend([offset_markers, offset_markers+1, offset_markers+2])
    
    for f in frames:
        f.traces = frame_trace_indices

    # Layout Settings
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
                          args=[None, dict(frame=dict(duration=1000/30*args.step, redraw=True), fromcurrent=True)]),
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
    
    print(f"Saving to {output_path}...")
    fig.write_html(output_path, auto_play=False)
    print("Done.")

if __name__ == "__main__":
    main()
