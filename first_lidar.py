import numpy as np
import matplotlib.pyplot as plt;
from load_data import load_lidar
from MAP import valid_range, lidar_map_init, lidar_pos
from model import motion_obs_model
from transforms import *
from pr2_utils import *

_, ranges = load_lidar()
angles_range = np.linspace(-5, 185, 286) / 180 * np.pi
lidar_scan, angles = valid_range(ranges[0, :], angles_range)
lidar_map = lidar_map_init()


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
    # # plot map
    plt.imshow(MAP['map'], cmap='gray')
    plt.title("Map")
    plt.pause(0.5)
    return MAP


def xy2map(x, y):
    """
    Transform the world frame x, y to the map frame cells(occupancy grid map)
    """
    MAP = lidar_map_init()
    xis = np.ceil((x - MAP['xmin']) / MAP['res']).astype(np.int64) - 1
    yis = np.ceil((y - MAP['ymin']) / MAP['res']).astype(np.int64) - 1
    return xis, yis


n_particles = 1
x_part, y_part, theta_part, weight = motion_obs_model().init_particles(n_particles)
updated_map = update_map(x_part[:, 0], y_part[:, 0], theta_part[:, 0], lidar_scan, angles, lidar_map)
plt.imshow(updated_map['map'], cmap='gray')
plt.pause(10)
