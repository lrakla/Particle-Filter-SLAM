
from transforms import *
import numpy as np
from pr2_utils import *
import cv2
from MAP import *
from load_data import load_stereo
from os.path import exists
from tqdm import tqdm


def get_stereo_parms():
    """
    Initialize stereo parameters
    """
    stereo = {}
    stereo["baseline"] = 475.143600050775 / 1000
    Ks = np.array([[8.1690378992770002e+02, 5.0510166700000003e-01,
                    6.0850726281690004e+02], [0., 8.1156803828490001e+02,
                                              2.6347599764440002e+02], [0., 0., 1.]])
    K_right = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02,
                         -3.6841758740842312e+02], [0., 7.7537235550066748e+02,
                                                    2.5718049049377441e+02, 0.], [0., 0., 1., 0.]])
    K_left = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0.], [0.,
                                                                                  7.7537235550066748e+02,
                                                                                  2.5718049049377441e+02, 0.],
                       [0., 0., 1.,
                        0.]])
    stereo["fsu"] = K_left[0, 0]
    stereo["fsv"] = K_left[1, 1]
    stereo["cu"] = K_left[0, 2]
    stereo["cv"] = K_left[1, 2]
    stereo["fsub"] = -K_right[0][3]
    return stereo


def get_world_coords_from_uv(disparity):
    """
    Convert pixel coordinates to vehicle coordinates using transforms and depth (calculated from disparity)
    """
    stereo = get_stereo_parms()
    fsu, fsv, cu, cv = stereo["fsu"], stereo["fsv"], stereo["cu"], stereo["cv"]
    z = stereo["fsub"] / (disparity + 1e-8)
    a = np.array(range(1280)).reshape(1, -1)
    b = np.array(range(560)).reshape(1, -1)
    uL = np.tile(a, (560, 1))
    vL = np.tile(b, (1280, 1)).T
    x = ((uL - cu) / fsu)*z
    y = ((vL - cv) / fsv) *z
    x_vec = x.flatten()
    y_vec = y.flatten()
    z_vec = z.flatten()
    sc = np.vstack((x_vec, y_vec, z_vec))
    return v2camera(sc)


def texture_map(MAP, x_part, y_part, theta_part, svc, image_l, texture_map):
    """
    Generate texture map using particle pose, pixel coordinates in vehicle frame and image
    """
    print(texture_map.shape)
    swc = w2v(x_part, y_part, theta_part, svc)  # every u,v has world frame coordinate
    z = swc[2, :]
    z[z > 2] = 0
    swc[2:] = z
    image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2RGB)  # 560x1280
    x_map, y_map = xy2map(swc[0, :], swc[1, :])
    x_map, y_map = x_map.reshape(560, 1280), y_map.reshape(560, 1280)
    indBad = np.logical_or(np.logical_or(np.logical_or((x_map < 1), (y_map < 1)), (x_map > MAP['sizex'])),
                           (y_map > MAP['sizey']))

    indheight = z == 0
    texture_map = image_l
    # set z above 50 to 0
    texture_map[indBad == 1] = (0, 0, 0)
    plt.imshow(texture_map)
    plt.pause(10)
    texture_map[indheight.reshape(560, 1280) == 1] = (0, 0, 0)
    plt.imshow(texture_map)
    plt.pause(10)
    return texture_map


if __name__ == '__main__':
    pass
    """
    Calculate disparity of all images and save in npy array
    """
    # stereo_time = load_stereo()
    # path_l = 'data/stereo_images/stereo_left'
    # path_r = 'data/stereo_images/stereo_right'
    # for time in tqdm(stereo_time):c
    #     path_r = 'data/stereo_images/stereo_right/' + time + '.png'
    #     if exists(path_r):
    #         path_l = 'data/stereo_images/stereo_left/' + time + '.png'
    #         disparity = compute_stereo(path_l, path_r)
    #         np.save(f'disparity/{time}.npy',disparity)
