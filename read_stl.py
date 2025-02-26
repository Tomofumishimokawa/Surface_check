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
