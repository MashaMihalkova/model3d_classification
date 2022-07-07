import numpy as np

from Mesh import Mesh


class Scene:
    def __init__(self, filename: str = ""):
        self.mesh_list: list = []
        self.filename: str = filename
        self.mesh_count: int = 0

    def __getitem__(self, item: int) -> Mesh:
        assert isinstance(item, int), "Only integer item."
        return self.mesh_list[item]

    def __str__(self):
        return f"mesh_list: {[ str(mesh) for mesh in self.mesh_list]}"

    def add_mesh(self, mesh: Mesh) -> None:
        mesh.faces -= np.min(mesh.faces)
        self.mesh_list.append(mesh)
        self.mesh_count += 1



