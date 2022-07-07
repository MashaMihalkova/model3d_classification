from multipledispatch import dispatch
import numpy as np


class Mesh:
    def __init__(self, name: str = "", group: str = "", material: str = ""):
        self.name: str = name
        self.vertices: np.ndarray = np.array([])
        self.vertex_normals: np.ndarray = np.array([])
        self.faces: np.ndarray = np.array([])
        self.group: str = group
        self.material: str = material

    def __str__(self):
        return f"name: '{self.name}'; vert: {self.vertices}; face: {self.faces}"

    def add_vertex(self, vertex: list) -> None:
        if self.vertices.size == 0:
            self.vertices = np.array([vertex])
        else:
            self.vertices = np.append(self.vertices, [vertex], axis=0)

    def add_vertex_normals(self, vertex_normals: list) -> None:
        if self.vertex_normals.size == 0:
            self.vertex_normals = np.array([vertex_normals])
        else:
            self.vertex_normals = np.append(self.vertex_normals, [vertex_normals], axis=0)

    def add_face(self, face: list) -> None:
        if self.faces.size == 0:
            self.faces = np.array([face])
        else:
            self.faces = np.append(self.faces, [face], axis=0)



