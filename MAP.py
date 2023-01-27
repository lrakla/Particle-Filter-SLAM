import numpy as np
import matplotlib.pyplot as plt;
from transforms import *

plt.ion()
from pr2_utils import bresenham2D


def lidar_pos(lidar_scan, angles):
    """
    lidar_scan : Lidar scan within valid ranges
    angles : Lidar angles within valid ranges

    This function converts lidar scan from polar to cartesian coordinates
    """
    xs0 = lidar_scan * np.cos(angles)
    ys0 = lidar_scan * np.sin(angles)
    return xs0, ys0


def valid_range(range, angles):
    """
    Returns range and angles of lidar scans within certain limits
    """
    indValid = np.logical_and((range < 70), (range > 2))
    range = range[indValid]  # valid scan readings
    angles = angles[indValid]
    return range, angles


def map_init():
    """
    Initialize map with specific paramters
    """
    MAP = {}
    MAP['res'] = 1  # meters
    MAP['xmin'] = -100  # meters
    MAP['ymin'] = -1200
    MAP['xmax'] = 1300
    MAP['ymax'] = 200
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)
    return MAP


def lidar_map_init():
    lidar_map = {}
    lidar_map['res'] = 0.1  # meters
    lidar_map['xmin'] = -50  # meters
    lidar_map['ymin'] = -50
    lidar_map['xmax'] = 50
    lidar_map['ymax'] = 50
    lidar_map['sizex'] = int(np.ceil((lidar_map['xmax'] - lidar_map['xmin']) / lidar_map['res'] + 1))  # cells
    lidar_map['sizey'] = int(np.ceil((lidar_map['ymax'] - lidar_map['ymin']) / lidar_map['res'] + 1))
    lidar_map['map'] = np.zeros((lidar_map['sizex'], lidar_map['sizey']), dtype=np.int8)
    return lidar_map


def update_map(x_part, y_part, theta_part, lidar_scan, angles, MAP):
    """
    x_part is x cell coordinate single particle with max weight
    returns updated map
    """
    sx, sy = xy2map(x_part, y_part)
    sx, sy = sx[0], sy[0]

    # end of lidar in lidar frame
    ex, ey = lidar_pos(lidar_scan, angles)

    # end point in 3d
    e_3d = np.ones((3, np.size(ex)))
    e_3d[0, :] = ex
    e_3d[1, :] = ey
    e_3d[2, :] = 0
    e_l2v = v2l(e_3d)
    e_l2w = w2v(x_part, y_part, theta_part, e_l2v)
    # valid_z = np.where(e_l2w[2, :] < 1.5)
    # ex, ey = xy2map(e_l2w[0, valid_z], e_l2w[1, valid_z])  # 1 x no of obstacles
    ex, ey = xy2map(e_l2w[0, :], e_l2w[1, :])

    # Use bresenham algorithm to detect how many grids that the lidar scan goes through
    for i in range(np.size(lidar_scan)):  # for each end point in lidar scan
        bresenham_pts = bresenham2D(sx, sy, ex[i], ey[i])
        bresenham_x = bresenham_pts[0, :].astype(np.int16)  # 1 x n_free_cells
        bresenham_y = bresenham_pts[1, :].astype(np.int16)
        indGood = np.logical_and(np.logical_and(np.logical_and((bresenham_x > 1), (bresenham_y > 1)),
                                                (bresenham_x < MAP['sizex'])), (bresenham_y < MAP['sizey']))
        ## Update Map
        # decrease log-odds if cell observed free
        MAP['map'][bresenham_x[indGood], bresenham_y[indGood]] = MAP['map'][bresenham_x[indGood], bresenham_y[
            indGood]] - np.log(4)
        # increase log-odds if cell observed occupied
        if ((ex[i] > 1) and (ex[i] < MAP['sizex']) and (ey[i] > 1) and (ey[i] < MAP['sizey'])):
            MAP['map'][ex[i], ey[i]] += np.log(4)
    return MAP


def log2binary(logoddsmap):
    """
    Transform the map odds value into 0, 1,0.7 (free,obstacle,unknown)
    """
    binary_map = np.zeros_like(logoddsmap)
    binary_map[logoddsmap < 0] = 0  # set free cell to black
    binary_map[logoddsmap > 0] = 1  # set occupied cell to white
    binary_map[logoddsmap == 0] = 0.7  # set unknown cell to grey
    return binary_map


def xy2map(x, y):
    """
    Transform the world frame x, y to the map frame cells(occupancy grid map)
    """
    MAP = map_init()
    xis = np.ceil((x - MAP['xmin']) / MAP['res']).astype(np.int64) - 1
    yis = np.ceil((y - MAP['ymin']) / MAP['res']).astype(np.int64) - 1
    return xis, yis


if __name__ == '__main__':
    pass
