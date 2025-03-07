def read_stl_text(file):
    verts = []
    faces = []
    vert_map = {}
    current_face = []

    for line in file:
        parts = line.strip().split()
        if parts[0] == 'vertex':
            vertex = tuple(float(p) for p in parts[1:])
            if vertex not in vert_map:
                vert_map[vertex] = len(verts)
                verts.append(vertex)
            current_face.append(vert_map[vertex])
            if len(current_face) == 3:
                faces.append(current_face)
                current_face = []

    return verts, faces
”””
!pip install trimesh matplotlib numpy scipy
!pip install rtree
!pip install shapely
"""
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# アップロードされたファイル名を取得
filename = "chair_0001.off"

# メッシュの読み込み
mesh = trimesh.load_mesh(filename)

# SDF 計算用のグリッド座標を生成
grid_size = 100  # SDF の解像度（増やすと精度が上がるが計算コストが増大）
x = np.linspace(mesh.bounds[0, 0], mesh.bounds[1, 0], grid_size)
y = np.linspace(mesh.bounds[0, 1], mesh.bounds[1, 1], grid_size)
z = np.linspace(mesh.bounds[0, 2], mesh.bounds[1, 2], grid_size)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# SDF 計算（符号付き距離関数）
sdf_values = trimesh.proximity.signed_distance(mesh, grid_points)
sdf_values = sdf_values.reshape((grid_size, grid_size, grid_size))

# Plotly で 3D Iso-surface を可視化
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=sdf_values.flatten(),
    isomin=-0.01,  # SDF の 0 付近を表示
    isomax=0.01,
    surface_count=3,  # いくつの等値面を描くか
    colorscale="viridis",
    opacity=0.6
))

fig.update_layout(title="3D SDF Visualization with Iso-surface")
fig.show()
