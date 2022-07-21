import numpy as np
import os
from obj_parser.Mesh import Mesh


class Scene:
    def __init__(self, filename: str = ""):
        self.mesh_list: list = []
        self.filename: str = filename
        self.mesh_count: int = 0
        self.material: str = ""

    def __getitem__(self, item: int) -> Mesh:
        assert isinstance(item, int), "Only integer item."
        return self.mesh_list[item]

    def __str__(self):
        return f"mesh_list: {[ str(mesh) for mesh in self.mesh_list]}"

    def add_mesh(self, mesh: Mesh) -> None:
        mesh.non_index_faces = mesh.faces.copy()
        mesh.faces -= np.min(mesh.faces)
        self.mesh_list.append(mesh)
        self.mesh_count += 1

    def save_to_obj(self, save_path: str) -> bool:
        assert os.path.exists(os.path.dirname(save_path)), "Save folder not exist"
        with open(save_path, 'w', newline='', encoding="utf-8") as f:
            f.write("# Pares by Igor\n# Made without love\n")
            f.write(f"mtllib {self.material}\n")
            for mesh in self.mesh_list:
                f.write(f'o {mesh.name}\n')
                for vertex in mesh.vertices:
                    f.write('v {:.6f} {:.6f} {:.6f}\n'.format(*vertex))
                for vertex_normals in mesh.vertex_normals:
                    f.write('vn {:.4f} {:.4f} {:.4f}\n'.format(*vertex_normals))
                f.write(f'g {mesh.group}\n')
                f.write(f'usemtl {mesh.material}\n')
                # f.write(f's {mesh.shading}\n')
                for face in mesh.non_index_faces:
                    f.write('f {} {} {}\n'.format(*face))
        return True



