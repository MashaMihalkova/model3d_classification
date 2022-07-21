import re
from obj_parser.Scene import Scene
# from Scene import Scene
from obj_parser.Mesh import Mesh

_debug = False


def read_obj(filename: str) -> Scene:
    scene = Scene(filename)
    mesh: Mesh = Mesh()
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.replace('\n', '')
            data = line.split(" ")
            if data[0] == 'o':
                if mesh.name:
                    scene.add_mesh(mesh)
                if _debug:
                    print("new MESH", data[1], scene.mesh_count)
                mesh = Mesh(data[1])
            elif data[0] == 'g':
                mesh.group = data[1]
            elif data[0] == 'usemtl':
                mesh.material = data[1]
            elif data[0] == 'v':
                mesh.add_vertex([float(elem) for elem in data[1:]])
            elif data[0] == 'vn':
                mesh.add_vertex_normals([float(elem) for elem in data[1:]])
            elif data[0] == 'f':
                face_vector = []
                for face in data[1:]:
                    res = re.findall(r'^(\d{1,10})//\d{1,10}$|^(\d{1,10})/\d{1,10}$|^(\d{1,10})$', face)
                    if res and any(res[0]):
                        for val in res[0]:
                            if val:
                                face_vector.append(int(val))
                                break
                mesh.add_face(face_vector)
        scene.add_mesh(mesh)
    return scene

