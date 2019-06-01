import numpy as np
import torch
import math

fovy = 0.5 
far, near = 1500, 250
 
top = math.tan(fovy/2) * near
right = top
bottom = -top
left = -top 


def rotationMatrix(theta_x, theta_y, theta_z):
    x = torch.Tensor([(1, 0, 0),
                        (0, torch.cos((theta_x*np.pi)/180), -torch.sin((theta_x*np.pi)/180)),
                        (0, torch.sin((theta_x*np.pi)/180), torch.cos((theta_x*np.pi)/180))])    
    y = torch.Tensor([(torch.cos((theta_y*np.pi)/180), 0, torch.sin((theta_y*np.pi)/180)),
                        (0, 1, 0),
                        (-torch.sin((theta_y*np.pi)/180), 0, torch.cos((theta_y*np.pi)/180))])    
    z = torch.Tensor([(torch.cos((theta_z*np.pi)/180), -torch.sin((theta_z*np.pi)/180), 0),
                        (torch.sin((theta_z*np.pi)/180), torch.cos((theta_z*np.pi)/180), 0),
                        (0, 0, 1)])    
    return z @ y @ x


def transformMatrix(thetas, translation):
    T = torch.eye(4)
    rotation = rotationMatrix(thetas[0], thetas[1], thetas[2])
    T[:3, :3] = rotation
    T[:3, 3] = translation    
    return T


def perspectiveProjectionMatrix(l,r,b,t,n,f): 
    P = torch.Tensor([[(2 * n) / (r - l), 0, (r + l) / (r - l), 0],
                      [0, (2 * n) / (t - b), (t + b) / (t - b), 0],
                      [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                      [0, 0, -1, 0]]) 
    return P


def viewportMatrix(l, r, b, t):  
    scal = torch.Tensor([(r - l) / 2, (t - b) / 2, 1 / 2])
    trans = torch.Tensor([(r + l) / 2, (t + b) / 2, 1 / 2])
 
    V = torch.eye(4)
    V[:3, 3] = trans 
    V[:3, :3] = torch.diag(scal) 
    return V


P = perspectiveProjectionMatrix(left,right,bottom,top,near,far)
V = viewportMatrix(left,right,bottom,top)   

def projection3Dto2D(G, w, t, V=V, P=P):    
    last_col = torch.ones((G.shape[0],1))
    G = torch.cat((G, last_col), 1)
       
    T = transformMatrix(w, t)
    
    proj = V @ P @ T
    proj = proj @ G.t()
    d = proj[3, :]
    proj = proj/d
    return proj.t()[:,0:2]



