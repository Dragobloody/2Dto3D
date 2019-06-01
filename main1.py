import os
import numpy as np
import trimesh
import pyrender
import h5py
import torch
from data_def import PCAModel, Mesh

bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
N_face = 30
N_exp = 20

# Facial Identity
face_mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
face_mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))
face_pcaBasis = np.asarray(bfm['color/model/pcaBasis'], dtype=np.float32).reshape((-1, 3, 199))
face_pcaVariance = np.asarray(bfm['color/model/pcaVariance'], dtype=np.float32)

# Expression Identity
exp_mean_shape = np.asarray(bfm['expression/model/mean'], dtype=np.float32).reshape((-1, 3))
exp_pcaBasis = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32).reshape((-1, 3, 100))
exp_pcaVariance = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)

# Triangle topology
face_triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
color_mean = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))

print('shape of Stuff:')
print('face_pcaBasis:', face_pcaBasis.shape)
print('face_pcaVariance:', face_pcaVariance.shape)
print('exp_pcaBasis:', exp_pcaBasis.shape)
print('exp_pcaVariance:', exp_pcaVariance.shape)
print('face_mean_shape:', face_mean_shape.shape)

mu_id = torch.from_numpy(face_mean_shape)
pca_basis_id = torch.from_numpy(face_pcaBasis[:, :, :N_face])
sigma_id = torch.from_numpy(np.sqrt(face_pcaVariance[:N_face]))

mu_exp = torch.from_numpy(exp_mean_shape)
pca_basis_exp = torch.from_numpy(exp_pcaBasis[:, :, :N_exp])
sigma_exp = torch.from_numpy(np.sqrt(exp_pcaVariance[:N_exp]))

model_id = PCAModel(mu_id, pca_basis_id, sigma_id)
model_exp = PCAModel(mu_exp, pca_basis_exp, sigma_exp)

alpha = torch.from_numpy(np.random.uniform(-1, 1, (N_face,))).type(torch.float32)
delta = torch.from_numpy(np.random.uniform(-1, 1, (N_exp,))).type(torch.float32)

E_id = model_id.pc @ (alpha * model_id.std).t()
E_exp = model_exp.pc @ (delta * model_exp.std).t()
G_id = model_id.mean + E_id
G_exp = model_exp.mean + E_exp
G = G_id + G_exp

def mesh_to_png(file_name, mesh):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)
    png = mesh.scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)


if __name__ == '__main__':
    #raise ValueError
    print('shape of face_triangles: ', face_triangles.shape)
    mesh = Mesh(G, color_mean, face_triangles)
    mesh_to_png("sample8.png", mesh)
