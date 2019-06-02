import sys
import os
import dlib
import glob
import torch
import cv2
from torch.autograd import Variable
from main1 import *
import numpy as np
from projection_3d_to_2d import *
import matplotlib.pyplot as plt

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_ground_truth(image_path, predictor_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    print("Processing file: {}".format(image_path))
    img = dlib.load_rgb_image(image_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        landmark_coords = shape_to_np(shape)

    ground_truth = torch.Tensor(landmark_coords)
    return ground_truth


def get_landmarks(landmarks_idx_file):
    # calculate our landmarks
    landmarks = np.fromfile(landmarks_idx_file, dtype=int, sep='\n')
    l_mu_id = mu_id[landmarks, :]
    l_pca_basis_id = pca_basis_id[landmarks, :, :]
    l_mu_exp = mu_exp[landmarks, :]
    l_pca_basis_exp = pca_basis_exp[landmarks, :, :]
    model_id = PCAModel(l_mu_id, l_pca_basis_id, sigma_id)
    model_exp = PCAModel(l_mu_exp, l_pca_basis_exp, sigma_exp)
    return model_id, model_exp


def L_lan(p, l):
    return torch.dist(p, l, p=2)


def L_reg(alpha, delta, lambda_alpha, lambda_delta):
    L_reg = lambda_alpha * torch.sum(alpha**2) + lambda_delta * torch.sum(delta**2)
    return L_reg




ground_truth = get_ground_truth('images4/dragos.jpg', 'shape_predictor_68_face_landmarks.dat')
ground_truth[:,0]=ground_truth[:,0]-200.0
ground_truth[:,1]=ground_truth[:,1]-200.0
model_id, model_exp = get_landmarks('Landmarks68_model2017-1_face12_nomouth.anl')

alpha = Variable(torch.zeros(N_face, ), requires_grad=True)
delta = Variable(torch.zeros(N_exp, ), requires_grad=True)
#w = Variable(torch.Tensor([0,0,0]), requires_grad=True)
w = Variable(torch.zeros((3, )), requires_grad=True)
#t = Variable(torch.ones(3, )*100, requires_grad=True)
t = Variable(torch.Tensor([0.0, 0.0, 100.0]), requires_grad=True)
lambda_alpha = 1000.0
lambda_delta = 1000.0
optimizer = torch.optim.Adam([alpha, delta, w, t], lr=0.01)
loss_list = []
max_iter = 10000

for i in range(max_iter):

    # Calculate
    E_id = model_id.pc @ (alpha * model_id.std).t()
    E_exp = model_exp.pc @ (delta * model_exp.std).t()
    G_id = model_id.mean + E_id
    G_exp = model_exp.mean + E_exp
    G = G_id + G_exp

    landmarks = projection3Dto2D(G, w, t)
    loss = L_lan(landmarks, ground_truth) + L_reg(alpha, delta, lambda_alpha, lambda_delta)
    loss_list.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        #print(f'alpha: {alpha}, delta: {delta}, w: {w}, t: {t}')
        print(f'Loss at iteration {i}: {loss}; '
              f'L_reg={L_reg(alpha, delta, lambda_alpha, lambda_delta)} L_lan={L_lan(landmarks, ground_truth)}')

plt.scatter(landmarks.detach().numpy()[:, 0], -landmarks.detach().numpy()[:,1])
plt.scatter(ground_truth.numpy()[:, 0], -ground_truth.numpy()[:, 1])
plt.show()
