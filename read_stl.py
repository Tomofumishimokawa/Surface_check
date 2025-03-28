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
