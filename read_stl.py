import numpy as np
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

