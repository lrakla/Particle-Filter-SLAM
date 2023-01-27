import numpy as np
from MAP import log2binary, lidar_pos, xy2map
from transforms import v2l, w2v
from pr2_utils import *


def get_velocity(tau, count_l, count_r):
    """
    Given change in encoder count and time, compute the linear velocity
    """
    resolution = 4096
    d_l = 0.623479
    d_r = 0.622806
    vl = np.pi * d_l * count_l / (resolution * tau)
    vr = np.pi * d_r * count_r / (resolution * tau)

    return (vl + vr) / 2


class motion_obs_model():
    def init_particles(self, n_particles):
        """
        Initialize particle pose to zeros
        """
        self.n_particles = n_particles
        # Initialize Particles
        self.x_part = np.zeros((1, self.n_particles))
        self.y_part = np.zeros((1, self.n_particles))
        self.theta_part = np.zeros((1, self.n_particles))
        self.weight = np.ones((1, self.n_particles)) / self.n_particles
        return self.x_part, self.y_part, self.theta_part, self.weight

    def predict_pose(self, x_part, y_part, theta_part, lin_velocity, delta_theta, tau, ADD_NOISE=False):
        """
        Predict pose of particle using previous pose and current velocity.
        Add Gaussian noise of 0 mean to x, y and theta
        """
        delta_x = lin_velocity * np.cos(theta_part) * tau
        delta_y = lin_velocity * np.sin(theta_part) * tau
        x_part += delta_x
        y_part += delta_y
        theta_part += delta_theta
        n_particles = x_part.shape[1]
        # add noise
        if ADD_NOISE:
            x_part = x_part + np.random.normal(0, abs(np.max(delta_x)) / 10, n_particles)
            y_part = y_part + np.random.normal(0, abs(np.max(delta_y)) / 10, n_particles)
            theta_part = theta_part + np.random.normal(0, abs(np.max(delta_theta)) / 10, n_particles)
        return x_part, y_part, theta_part

    def softmax(self, x):
        """
        Compute softmax vector of given x
        """
        e_x = np.exp(x)
        return e_x / np.sum(e_x)

    def update_pose(self, x_part, y_part, theta_part, weight, lidar_scan, angles, n_particles, MAP):
        """
        Update weights of all particles by using mapCorrelation
        Choose a 9x9 grid around a particle location, and find location of max correlation
        Update and normalize the new weights
        """
        binary_map = log2binary(MAP["map"])
        x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x index of each pixel on log-odds map
        y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y index of each pixel on log-odds map

        # 9x9 grid around particle with the particle being in the center
        xs = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # x deviation
        ys = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # y deviation

        ex, ey = lidar_pos(lidar_scan, angles)
        # end point in 3d
        e_3d = np.ones((3, np.size(ex)))
        e_3d[0, :] = ex
        e_3d[1, :] = ey
        e_l2v = v2l(e_3d)
        corr = np.zeros(n_particles)
        for i in range(n_particles):
            e_l2w = w2v(x_part[:, i], y_part[:, i], theta_part[:, i], e_l2v)  # 3xn
            valid_z = np.where(e_l2w[2, :] < 1.5)
            ex, ey = xy2map(e_l2w[0, valid_z], e_l2w[1, valid_z])  # 1 x no of obstacles
            vp = np.stack((ex, ey))  # 2 x no of obstacles
            # calculate correlation
            c = mapCorrelation(binary_map, x_im, y_im, vp, xs, ys)
            # find largest correlation
            corr[i] = np.max(c)

        # update particle weight with softmax function
        soft = self.softmax(corr)
        weight *= soft / np.sum(weight * soft)
        return weight

    def resampling(self, x_part, y_part, theta_part, weight, n_particles):
        """
        Stratified Resampling
        Returns resampled particle poses and weights
        """
        pose_part = np.vstack((x_part, y_part, theta_part))
        pose_part_updated = np.zeros((3, n_particles))
        weight_updated = np.tile(1 / n_particles, n_particles).reshape(1, n_particles)
        j = 0  # 1st particle has index 0
        c = weight[0, 0]  # weight of 1st particle

        for i in range(n_particles):
            u = np.random.uniform(0, 1 / n_particles)
            b = u + i / n_particles  # k-1 => i as k starts from 1
            while b > c:
                j += 1
                c += weight[0, j]
            # add to the new set
            pose_part_updated[:, i] = pose_part[:, j]

        return pose_part_updated[0, :], pose_part_updated[1, :], pose_part_updated[2, :], weight_updated
