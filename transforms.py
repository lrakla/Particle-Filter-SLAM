import numpy as np

"""
l : lidar 
v: vehicle a.k.a body a.k.a robot
f : FOG 
c : camera 
"""


def v2l(sl):
    """
    Lidar to vehicle
    """
    R = np.asarray([[0.00130201, 0.796097, 0.605167], [0.999999, -0.000419027, -0.00160026], [
        -0.00102038, 0.605169, -0.796097]])
    p = np.array([[0.8349, -0.0126869, 1.76416]]).T
    sl = np.concatenate((sl, np.ones((1, sl.shape[1]))), axis=0)
    T = np.concatenate((R, p), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    svl = np.dot(T, sl)
    return svl[:3, :]


def v2f(sf):
    """
    Fog to vehicle (not used)
    """
    sf = np.concatenate((sf, np.ones((1, sf.shape[1]))), axis=0)
    R = np.asarray([[1, 0, 0], [0, 1, 0], [
        0, 0, 1]])
    p = np.array([[-0.335, -0.035, 0.78]]).T
    T = np.concatenate((R, p), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    svf = np.dot(T, sf)
    return svf[:3, :]


def w2v(x, y, theta, sv):
    """
    Vehicle to world
    x: x position of particle
    y: y position of particle
    theta : theta of prticle
    sv : point in vehicle frame 3xn
    """
    sv = np.concatenate((sv, np.ones((1, sv.shape[1]))), axis=0)
    T = np.array([[np.cos(theta), -np.sin(theta), 0, x], [np.sin(theta), np.cos(theta), 0, y], \
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    swv = np.dot(T, sv)
    return swv[:3, :]


def v2camera(sc):
    """
    Camera to vehicle
    """
    R = np.asarray([[-0.00680499, -0.0153215, 0.99985], [-0.999977, 0.000334627, -0.00680066], [
        -0.000230383, -0.999883, -0.0153234]])
    R_inv = np.linalg.inv(R)
    p = np.array([[1.64239, 0.247401, 1.58411]]).T
    svc = np.dot(R_inv, sc) + p
    return svc
