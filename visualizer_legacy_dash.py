import dash
from dash import dcc, html, Input, Output, State, clientside_callback
import plotly.graph_objects as go
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from solvers.base_solver import OrientationTrajectory

def get_phone_mesh_initial():
    """Returns the initial Mesh3d traces (identity rotation) for the figure setup."""
    # Phone dimensions
    width, height, depth = 1.0, 2.0, 0.2
    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * (width / 2)
    y = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * (height / 2)
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * (depth / 2)
    
    # Body (Black)
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
    trace_body = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='#333333', name='Phone Body', showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2)
    )
    
    # Screen (Blue) - Offset slightly in Z+
    screen_idx = [0, 1, 2, 3]
    sx = x[screen_idx]
    sy = y[screen_idx]
    sz = z[screen_idx] + 0.02
    
    trace_screen = go.Mesh3d(
        x=sx, y=sy, z=sz,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#00AAFF', opacity=0.9, name='Screen', showscale=False
    )
    
    # Camera Bump (Red) - Back Z-
    # Center approx (0, 0.6, -0.1)
    bx = np.array([-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3]) 
    by = np.array([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9])
    bz = np.array([-0.1, -0.1, -0.1, -0.1, -0.15, -0.15, -0.15, -0.15]) # Protrudes backwards 
    
    trace_bump = go.Mesh3d(
        x=bx, y=by, z=bz,
        i=i, j=j, k=k,
        color='#FF0000', name='Camera Bump', showscale=False,
        lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0) # Flat bright red
    )

    # Fixed Axes (East=Red, North=Green, Up=Blue)
    # Length 1.5
    trace_east = go.Scatter3d(
        x=[0, 1.5], y=[0, 0], z=[0, 0],
        mode='lines+text', line=dict(color='red', width=5),
        text=['', 'East'], textposition="top center", name='East'
    )
    trace_north = go.Scatter3d(
        x=[0, 0], y=[0, 1.5], z=[0, 0],
        mode='lines+text', line=dict(color='green', width=5),
        text=['', 'North'], textposition="top center", name='North'
    )
    trace_up = go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 1.5],
        mode='lines+text', line=dict(color='blue', width=5),
        text=['', 'Up'], textposition="top center", name='Up'
    )
    
    return [trace_body, trace_screen, trace_bump, trace_east, trace_north, trace_up]

def quat2rpy(quats):
    r = R.from_quat(quats[:, [1, 2, 3, 0]])
    return r.as_euler('xyz', degrees=True)

class Visualizer:
    def __init__(self, trajectory: OrientationTrajectory, video_path: str, gt_quats: np.ndarray = None):
        self.trajectory = trajectory
        self.video_path = video_path
        self.gt_quats = gt_quats

        
        # Determine assets folder relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_dir, 'assets')
        
        self.app = dash.Dash(__name__, assets_folder=assets_dir)
        
        self.ts = self.trajectory.timestamps
        self.quats = self.trajectory.quaternions # [w, x, y, z]
        self.covs = self.trajectory.covariances
        self.dt = np.mean(np.diff(self.ts)) if len(self.ts) > 1 else 0.02
        
        self.rpy = quat2rpy(self.quats)
        self.sigmas = np.sqrt(np.diagonal(self.covs, axis1=1, axis2=2)) * 180.0 / np.pi
        
        if self.gt_quats is not None:
            self.gt_rpy = quat2rpy(self.gt_quats)
        else:
            self.gt_rpy = None

        
        self._setup_layout()
        self._setup_clientside_callbacks()
        
    def _setup_layout(self):
        # Prepare data for client-side stores
        timestamps_list = self.ts.tolist()
        quats_list = self.quats.tolist() 
        
        initial_traces = get_phone_mesh_initial()
        
        layout_3d = go.Layout(
            scene=dict(
                xaxis=dict(range=[-2, 2], title='', showticklabels=False),
                yaxis=dict(range=[-2, 2], title='', showticklabels=False),
                zaxis=dict(range=[-2, 2], title='', showticklabels=False),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False
        )
        
        # Static RPY Plot
        fig_rpy = go.Figure()
        
        def add_band(fig, x, y_mean, sigma, color, name):
            upper = y_mean + 2.0*sigma
            lower = y_mean - 2.0*sigma
            # Use Scattergl for performance
            fig.add_trace(go.Scattergl(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor=color, opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{name} 2σ'
            ))
            fig.add_trace(go.Scattergl(x=x, y=y_mean, line=dict(color=color), name=name))

        add_band(fig_rpy, self.ts, self.rpy[:,0], self.sigmas[:,0], 'red', 'Roll')
        add_band(fig_rpy, self.ts, self.rpy[:,1], self.sigmas[:,1], 'green', 'Pitch')
        add_band(fig_rpy, self.ts, self.rpy[:,2], self.sigmas[:,2], 'blue', 'Yaw')
        
        # Add GT if available
        if self.gt_rpy is not None:
             # Ensure lengths match or crop
             min_len = min(len(self.ts), len(self.gt_rpy))
             # Roll GT
             fig_rpy.add_trace(go.Scattergl(
                 x=self.ts[:min_len], y=self.gt_rpy[:min_len, 0],
                 mode='lines', line=dict(color='darkred', width=1, dash='dot'),
                 name='GT Roll'
             ))
             # Pitch GT
             fig_rpy.add_trace(go.Scattergl(
                 x=self.ts[:min_len], y=self.gt_rpy[:min_len, 1],
                 mode='lines', line=dict(color='darkgreen', width=1, dash='dot'),
                 name='GT Pitch'
             ))
             # Yaw GT
             fig_rpy.add_trace(go.Scattergl(
                 x=self.ts[:min_len], y=self.gt_rpy[:min_len, 2],
                 mode='lines', line=dict(color='darkblue', width=1, dash='dot'),
                 name='GT Yaw'
             ))

        
        # Current Time Line (Trace 6)
        fig_rpy.add_trace(go.Scattergl(
            x=[self.ts[0], self.ts[0]], y=[-360, 360],
            mode='lines', line=dict(color='black', width=2, dash='dash'),
            name='Current Time'
        ))
        
        fig_rpy.update_layout(
             margin=dict(l=40, r=40, b=30, t=10),
             yaxis_title="Degrees", xaxis_title="Time (s)",
             yaxis=dict(range=[-200, 200]), 
             hovermode="x unified"
        )

        self.app.layout = html.Div([
            html.H2("Phone Orientation - FINAL FIXED VERSION", style={'textAlign': 'center', 'color': 'red'}),
            
            html.Div([
                # Left Column
                html.Div([
                    html.H4("Synced Video"),
                    html.Video(
                        id='video-player',
                        src="/assets/video.mp4",
                        controls=True,
                        style={'width': '300px', 'maxWidth': '100%', 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto'}
                    ),
                    html.Div([
                        html.Button('Play/Pause', id='play-btn', n_clicks=0, style={'marginRight': '10px'}),
                        html.Label("Speed:", style={'marginLeft': '10px', 'marginRight': '5px'}),
                        dcc.Dropdown(
                            id='speed-dropdown',
                            options=[
                                {'label': '0.5x (Slow)', 'value': 2.0},
                                {'label': '1.0x (Real-time)', 'value': 1.0},
                                {'label': '2.0x (Fast)', 'value': 0.5},
                                {'label': '5.0x (Super Fast)', 'value': 0.2}
                            ],
                            value=1.0,
                            clearable=False,
                            style={'width': '150px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                        )
                    ], style={'marginTop': '10px'}),
                    
                    html.H4("Synchronization"),
                    html.Label("Video Offset (s) - Use to sync video to motion:"),
                    dcc.Slider(
                        min=-5.0, max=5.0, step=0.1, value=0.0,
                        id='sync-slider',
                        marks={i: str(i) for i in range(-5, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.H4("Navigation"),
                    dcc.Slider(
                        min=0, max=len(self.ts)-1, step=1, value=0,
                        id='time-slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                    
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                
                # Right Column
                html.Div([
                    html.H4("Real-time Orientation"),
                    dcc.Graph(
                        id='3d-plot',
                        figure=go.Figure(data=initial_traces, layout=layout_3d),
                        style={'height': '40vh'}
                    ),
                    html.H4("Orientation Angles (2σ)"),
                    dcc.Graph(id='rpy-plot', figure=fig_rpy, style={'height': '35vh'})
                ], style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
            ]),
            
            # Stores for Client-side Data
            dcc.Store(id='store-timestamps', data=timestamps_list),
            dcc.Store(id='store-quats', data=quats_list),
            # Interval drives the loop - minimum 33ms (30fps)
            dcc.Interval(id='interval-component', interval=max(33, int(self.dt*1000)), n_intervals=0, disabled=True)
            
        ], style={'fontFamily': 'sans-serif'})

    def _setup_clientside_callbacks(self):
        # 1. Main Sync Callback (Slider + Sync Offset -> Updates 3D Mesh + Video + RPY Line)
        self.app.clientside_callback(
            """
            function(idx, offset, timestamps, quats, fig3d, figRPY) {
                if (!timestamps || !quats || !fig3d) return window.dash_clientside.no_update;
                
                var t = timestamps[idx];
                var q = quats[idx]; // [w, x, y, z]
                
                // Sync Video: Apply offset. 
                // t is data time. video needs to be at t - offset? 
                // No, if Offset > 0, Video Leans Forward? 
                // Let's define offset means: Video Time = Data Time + Offset
                var vidTime = t + offset;
                var vid = document.getElementById('video-player');
                
                if (vid && vidTime >= 0 && Math.abs(vid.currentTime - vidTime) > 0.15) {
                    vid.currentTime = vidTime;
                }

                // --- 3D Rotation Math (Client-side) ---
                var w = 1.0, h = 2.0, d = 0.2;
                var qw = q[0], qx = q[1], qy = q[2], qz = q[3];
                // Quaternion to Matrix
                var xx = qx*qx, yy = qy*qy, zz = qz*qz;
                var xy = qx*qy, xz = qx*qz, yz = qy*qz;
                var wx = qw*qx, wy = qw*qy, wz = qw*qz;
                
                var m00 = 1 - 2*(yy+zz), m01 = 2*(xy-wz),   m02 = 2*(xz+wy);
                var m10 = 2*(xy+wz),     m11 = 1 - 2*(xx+zz), m12 = 2*(yz-wx);
                var m20 = 2*(xz-wy),     m21 = 2*(yz+wx),     m22 = 1 - 2*(xx+yy);
                
                function rotate(xArr, yArr, zArr) {
                    var xr = [], yr = [], zr = [];
                    for(var i=0; i<xArr.length; i++) {
                        xr.push(m00*xArr[i] + m01*yArr[i] + m02*zArr[i]);
                        yr.push(m10*xArr[i] + m11*yArr[i] + m12*zArr[i]);
                        zr.push(m20*xArr[i] + m21*yArr[i] + m22*zArr[i]);
                    }
                    return [xr, yr, zr];
                }
                
                // 1. Body
                var bx = [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5].map(v => v * w);
                var by = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5].map(v => v * h);
                var bz = [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5].map(v => v * d);
                var bodyRot = rotate(bx, by, bz);
                
                // 2. Screen (Offset transformed z-normal)
                var offX = m02 * 0.02, offY = m12 * 0.02, offZ = m22 * 0.02;
                var screenIndices = [0,1,2,3];
                var sx = [], sy = [], sz = [];
                screenIndices.forEach(i => {
                    sx.push(bodyRot[0][i] + offX);
                    sy.push(bodyRot[1][i] + offY);
                    sz.push(bodyRot[2][i] + offZ);
                });
                
                // 3. Camera Bump (Centered at y=0.65, z=-0.1...-0.15)
                // Default coords
                var cx = [-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3];
                var cy = [0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9];
                var cz = [-0.1, -0.1, -0.1, -0.1, -0.15, -0.15, -0.15, -0.15];
                var bumpRot = rotate(cx, cy, cz);
                
                // Update Figure
                var newFig3d = Object.assign({}, fig3d);
                newFig3d.data = [...fig3d.data]; // Shallow copy array
                
                // Update Body
                newFig3d.data[0].x = bodyRot[0];
                newFig3d.data[0].y = bodyRot[1];
                newFig3d.data[0].z = bodyRot[2];
                
                // Update Screen
                newFig3d.data[1].x = sx;
                newFig3d.data[1].y = sy;
                newFig3d.data[1].z = sz;
                
                // Update Bump
                if (newFig3d.data[2]) {
                    newFig3d.data[2].x = bumpRot[0];
                    newFig3d.data[2].y = bumpRot[1];
                    newFig3d.data[2].z = bumpRot[2];
                }
                
                var newFigRPY = Object.assign({}, figRPY);
                newFigRPY.data = [...figRPY.data];
                
                // Trace 6 is the vertical line
                if (newFigRPY.data.length > 6) {
                    newFigRPY.data[6].x = [t, t];
                }
                
                return [newFig3d, newFigRPY];
            }
            """,
            [Output('3d-plot', 'figure'), Output('rpy-plot', 'figure')],
            [Input('time-slider', 'value'), Input('sync-slider', 'value')],
            [State('store-timestamps', 'data'),
             State('store-quats', 'data'),
             State('3d-plot', 'figure'),
             State('rpy-plot', 'figure')]
        )

        # 2. Controls: Play/Pause
        @self.app.callback(
             Output('interval-component', 'disabled'),
             [Input('play-btn', 'n_clicks')],
             [State('interval-component', 'disabled')]
        )
        def toggle(n, dis):
            if n: return not dis
            return True # Start Disabled (Paused)
        
        # 3. Animation Loop (Interval -> Slider)
        @self.app.callback(
            Output('time-slider', 'value'),
            [Input('interval-component', 'n_intervals')],
            [State('time-slider', 'value'), State('time-slider', 'max')]
        )
        def advance(n, val, max_val):
            if val < max_val: return val + 1
            return 0
            
        # 4. Speed Control
        @self.app.callback(
            Output('interval-component', 'interval'),
            [Input('speed-dropdown', 'value')]
        )
        def set_speed(multiplier):
            # Target 30Hz or so
            base = max(33, int(self.dt * 1000))
            return int(base * multiplier)

    def save_html(self, output_path: str = "orientation_viz.html"):
        print(f"Generating HTML: {output_path}...")
        frames = []
        step = 1 if len(self.ts) < 1000 else len(self.ts) // 500
        for i in range(0, len(self.ts), step):
            data = get_phone_mesh(self.quats[i])
            frames.append(go.Frame(data=data, name=str(self.ts[i])))
            
        fig = go.Figure(
            data=get_phone_mesh(self.quats[0]),
            layout=go.Layout(
                title="Phone Orientation (2σ)",
                scene=dict(aspectmode='cube', xaxis=dict(range=[-2,2]), yaxis=dict(range=[-2,2]), zaxis=dict(range=[-2,2])),
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
                    ],
                    "type": "buttons"
                }],
                sliders=[{
                    "steps": [{"args": [[str(self.ts[i])], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}], "label": f"{self.ts[i]:.1f}", "method": "animate"} for i in range(0, len(self.ts), step)]
                }]
            ),
            frames=frames
        )
        fig.write_html(output_path)
        print("HTML saved.")

    def run(self):
        # self.save_html() 
        print("Starting Optimized Dash Server on Port 8051...")
        self.app.run(debug=True, use_reloader=False, port=8051)

def get_phone_mesh(q):
    width, height, depth = 1.0, 2.0, 0.2
    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * (width / 2)
    y = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * (height / 2)
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * (depth / 2)
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    verts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    verts_rot = r.apply(verts)
    x_rot, y_rot, z_rot = verts_rot[:, 0], verts_rot[:, 1], verts_rot[:, 2]
    
    trace_body = go.Mesh3d(
        x=x_rot, y=y_rot, z=z_rot,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color='#333333', name='Phone Body', showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2)
    )
    
    normal = r.apply(np.array([0, 0, 1]))
    offset = 0.02 * normal
    screen_idx = [0, 1, 2, 3]
    sx = x_rot[screen_idx] + offset[0]
    sy = y_rot[screen_idx] + offset[1]
    sz = z_rot[screen_idx] + offset[2]
    
    trace_screen = go.Mesh3d(
        x=sx, y=sy, z=sz,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#00AAFF', opacity=0.9, name='Screen', showscale=False
    )
    
    # Static Export Bump
    bx = np.array([-0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3, -0.3])
    by = np.array([0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.9])
    bz = np.array([-0.1, -0.1, -0.1, -0.1, -0.15, -0.15, -0.15, -0.15])
    
    verts_bump = np.stack([bx.flatten(), by.flatten(), bz.flatten()], axis=1)
    rot_bump = r.apply(verts_bump)
    
    trace_bump = go.Mesh3d(
        x=rot_bump[:,0], y=rot_bump[:,1], z=rot_bump[:,2],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color='#FF0000', name='Camera Bump',
        lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0)
    )
    
    # We could add axes to HTML export too, but usually it's static frames.
    # Let's add them for consistency if useful, but static export might just focus on phone.
    # Adding simplified axes lines to export frames:
    
    trace_east = go.Scatter3d(
        x=[0, 1.5], y=[0, 0], z=[0, 0],
        mode='lines+text', line=dict(color='red', width=5),
        text=['', 'East'], textposition="top center"
    )
    trace_north = go.Scatter3d(
        x=[0, 0], y=[0, 1.5], z=[0, 0],
        mode='lines+text', line=dict(color='green', width=5),
        text=['', 'North'], textposition="top center"
    )
    trace_up = go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, 1.5],
        mode='lines+text', line=dict(color='blue', width=5),
        text=['', 'Up'], textposition="top center"
    )
    
    return [trace_body, trace_screen, trace_bump, trace_east, trace_north, trace_up]
