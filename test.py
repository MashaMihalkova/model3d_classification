import numpy as np
import os
import open3d as o3d
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data
from config import get_test_config
from data import ModelNet40
from models import MeshNet
from obj_parser.parser import *
from obj_parser import *
import pymeshlab
from utils.retrival import append_feature, calculate_map

# CLASS_NAMES = ["profile_beam", "pipes", "clamps_1", "clamps_2", "clamps_3", "handrails", "ladder", "stairs",
#                "ladder_cage", "profile_pipe", "profile_corner"]
# CLASS_NAMES_RUS = ["Профиль", "Труба", "Крепёж с петелькой", "Крепёж прикольный", "Крепёж труба",
#                    "Забор для рук", "Длинная лестница вверх", "Лестница",
#                    "Ограда для цветов", "Профильная труба", "Уголок"]
type_to_index_map = {
    "l_beam":0, "pipes":1, "clamps":2, "handrails":3, "ladder":4, "stairs":5, "ladder_cage":6, "platforms":7,"profile":8
}
CLASS_NAMES = ["l_beam", "pipes", "clamps", "handrails", "ladder", "stairs",
               "ladder_cage", "platforms", "profile"]
CLASS_NAMES_RUS = ["Профиль", "Труба", "Крепёж", "Забор для рук", "Длинная лестница вверх", "Лестница",
                   "Ограда для цветов", "Платформа", "Профиль"]
cfg = get_test_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = ModelNet40(cfg=cfg['dataset'], part='test')
data_loader = data.DataLoader(data_set, batch_size=cfg['batch_size'], num_workers=0, shuffle=True)


def test_model(model):

    correct_num = 0
    ft_all, lbl_all = None, None

    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
            centers = centers.cuda()
            corners = corners.cuda()
            normals = normals.cuda()
            neighbor_index = neighbor_index.cuda()
            targets = targets.cuda()

            outputs, feas = model(centers, corners, normals, neighbor_index)
            _, preds = torch.max(outputs, 1)

            correct_num += (preds == targets).float().sum()

            if cfg['retrieval_on']:
                ft_all = append_feature(ft_all, feas.detach().cpu())
                lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

    print('Accuracy: {:.4f}'.format(float(correct_num) / len(data_set)))
    if cfg['retrieval_on']:
        print('mAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def test_full_model(model, input_path):
    scene = read_obj(input_path)

    # print(classifier(np.random.rand(32**3)))

    for i in range(scene.mesh_count):
        mesh_obj = scene[i]

        if mesh_obj.name[:4] in ['Лист', 'Плит', 'Каме']:
            continue

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_obj.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_obj.faces)
        o3d.io.write_triangle_mesh("data_pred_full/ob1.obj", mesh)

        ms = pymeshlab.MeshSet()
        ms.clear()
        # load mesh
        ms.load_new_mesh('data_pred_full/ob1.obj')
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        max_faces = faces.shape[0]
        if faces.shape[0] > max_faces:
            print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], out_dir))
            continue

        # move to center
        center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
        vertices -= center

        # normalize
        max_len = np.max(vertices[:, 0] ** 2 + vertices[:, 1] ** 2 + vertices[:, 2] ** 2)
        vertices /= np.sqrt(max_len)

        # get normal vector
        ms.clear()
        mesh = pymeshlab.Mesh(vertices, faces)
        ms.add_mesh(mesh)
        face_normal = ms.current_mesh().face_normal_matrix()

        # get neighbors
        faces_contain_this_vertex = []
        for i in range(len(vertices)):
            faces_contain_this_vertex.append(set([]))
        centers = []
        corners = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            x1, y1, z1 = vertices[v1]
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]
            centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
            corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        neighbors = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])

        centers = np.array(centers)
        corners = np.array(corners)
        faces = np.concatenate([centers, corners, face_normal], axis=1)
        neighbors = np.array(neighbors)
        np.savez('data_pred_full/pipes/ob1.npz', faces=faces, neighbors=neighbors)

        data_set_ = ModelNet40(cfg=cfg['dataset_full'], part='test')
        data_loader_ = data.DataLoader(data_set_, batch_size=cfg['batch_size'], num_workers=0, shuffle=True)
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader_):
            pred, feas = model(centers, corners, normals, neighbor_index)
            # face = torch.tensor(faces).float()

            # neighbor_index = torch.from_numpy(neighbor_index).long()
            # neighbor_index = torch.tensor(neighbors).long()
            # target = torch.tensor(type, dtype=torch.long)

            # reorganize
            # face = face.permute(1, 0).contiguous()
            # centers, corners, normals = face[:3], face[3:12], face[12:]
            # corners = corners - torch.cat([centers, centers, centers], 0)
            # pred, feas = model(centers, corners, normals, neighbor_index)
            # pred = model(mesh)
            print(pred)
            if torch.max(pred) < 0.1:
                mesh_obj.name = "Без понятия что это"
            else:
                print(CLASS_NAMES_RUS[torch.argmax(pred)])
                mesh_obj.name = CLASS_NAMES_RUS[torch.argmax(pred)]

    scene.save_to_obj('D:\\work2\\classification_3d_models\\model_copy\\model_copy\\data\\test\\full_test_rename_cls11_lim400.obj')


if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    # model.cuda()
    # model = nn.DataParallel(model)
    k = torch.load(cfg['load_model'], map_location=torch.device('cpu'))
    model.load_state_dict(k, strict=False)
    model.eval()
    input_path = 'D:\\work2\\classification_3d_models\\model_copy\\model_copy\\data\\test\\full_test.obj'
    test_full_model(model, input_path)
