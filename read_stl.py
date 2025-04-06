import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_stl_text(file_path):
    """
    Reads an ASCII STL file and extracts vertex coordinates and face indices.

    Args:
        file_path (str): Path to the STL file.

    Returns:
        tuple: A tuple containing:
            - vertices (list of tuples): A list of unique vertex coordinates (x, y, z).
            - faces (list of lists): A list of faces, each represented as a list of three vertex indices.
    """
    verts = []  # List to store unique vertex coordinates
    faces = []  # List to store faces (triangles as indices)
    vert_map = {}  # Dictionary to map vertex coordinates to their index
    current_face = []  # Temporary list to store the current face's vertices

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'vertex':
                vertex = tuple(float(p) for p in parts[1:])
                if vertex not in vert_map:
                    vert_map[vertex] = len(verts)
                    verts.append(vertex)
                current_face.append(vert_map[vertex])
                
                if len(current_face) == 3:  # A face consists of exactly 3 vertices
                    faces.append(current_face)
                    current_face = []  # Reset for the next face

    return verts, faces

def prepare_stl_data_for_ISAB(file_names, num_points=5000, device='cuda'):
    """
    Loads and preprocesses point cloud data from STL files for ISAB-based Set-VAE.
    
    Args:
        file_names (list of str): List of file paths to STL files.
        num_points (int): Number of points to sample from each point cloud.
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        DataLoader: DataLoader object containing the preprocessed set-structured data.
    """
    input_data_list = []
    for file_name in file_names:
        verts, _ = read_stl_text(file_name)  # 修正: ファイルパスを渡す
        verts = np.array(verts)  # リスト → NumPy 配列へ変換
        
        if len(verts) == 0:
            print(f"Warning: {file_name} has no vertices.")
            continue

        # 点群サンプリング
        if len(verts) > num_points:
            sampled_indices = np.random.choice(len(verts), num_points, replace=False)
            pointcloud = verts[sampled_indices]
        else:
            pointcloud = verts  # 点が少ない場合、そのまま使用

        # 正規化（各次元ごとに [0,1] スケール）
        pointcloud = (pointcloud - pointcloud.min(axis=0)) / (pointcloud.max(axis=0) - pointcloud.min(axis=0) + 1e-8)
        
        input_data_list.append(pointcloud)

    if len(input_data_list) == 0:
        raise ValueError("No valid point cloud data found.")

    # データをスタック (num_samples, num_points, 3)
    input_data = np.stack(input_data_list)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)

def prepare_data_from_csv(file_names, num_points=5000, device='cuda'):
    """
    CSVファイルから点群を読み取り、均等サンプリング・正規化し、DataLoaderとして返す。
    Args:
        file_names (list of str): CSVファイルパスのリスト
        num_points (int): 各ファイルからのサンプル数
        device (str): 'cuda' または 'cpu'
    Returns:
        DataLoader: 点群テンソルのDataLoader
    """
    input_data_list = []

    for file_name in file_names:
        print(f"Processing file: {file_name}")
        df = pd.read_csv(file_name)

        # 必須カラムの存在確認
        if not all(col in df.columns for col in ["X (m)", "Y (m)", "Z (m)"]):
            print(f"Warning: Missing necessary columns in {file_name}. Skipping.")
            continue

        coords = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()

        # 均等グリッド生成：立方体グリッドを想定
        num_samples_per_axis = int(np.cbrt(num_points))  # 例: 5000 -> 約17
        x_vals = np.linspace(coords[:, 0].min(), coords[:, 0].max(), num_samples_per_axis)
        y_vals = np.linspace(coords[:, 1].min(), coords[:, 1].max(), num_samples_per_axis)
        z_vals = np.linspace(coords[:, 2].min(), coords[:, 2].max(), num_samples_per_axis)

        # 各セルに対応する点を選ぶ
        filtered_points = []
        for x in range(num_samples_per_axis - 1):
            for y in range(num_samples_per_axis - 1):
                for z in range(num_samples_per_axis - 1):
                    x_min, x_max = x_vals[x], x_vals[x + 1]
                    y_min, y_max = y_vals[y], y_vals[y + 1]
                    z_min, z_max = z_vals[z], z_vals[z + 1]

                    cell_points = coords[
                        (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
                        (coords[:, 1] >= y_min) & (coords[:, 1] < y_max) &
                        (coords[:, 2] >= z_min) & (coords[:, 2] < z_max)
                    ]

                    if len(cell_points) > 0:
                        sampled = cell_points[np.random.choice(len(cell_points))]
                        filtered_points.append(sampled)

        if len(filtered_points) == 0:
            print(f"Warning: No data after sampling in {file_name}. Skipping.")
            continue

        # 必要数に満たない場合、追加ランダムサンプリングで補完
        while len(filtered_points) < num_points:
            idx = np.random.randint(0, coords.shape[0])
            filtered_points.append(coords[idx])
        filtered_points = np.array(filtered_points[:num_points])

        # 正規化（0-1）
        pointcloud = (filtered_points - filtered_points.min(axis=0)) / (
            filtered_points.max(axis=0) - filtered_points.min(axis=0)
        )

        input_data_list.append(pointcloud)

    if len(input_data_list) == 0:
        raise ValueError("No valid data found in the provided CSV files.")

    input_data = np.stack(input_data_list)  # shape: (batch, N, 3)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)
