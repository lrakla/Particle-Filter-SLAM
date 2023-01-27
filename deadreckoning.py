import numpy as np
from MAP import update_map, map_init, valid_range
from pr2_utils import *
from model import motion_obs_model, get_velocity
import matplotlib.pyplot as plt
from load_data import load_encoder, load_lidar
from tqdm import tqdm

lidar_time, lidar_scans = load_lidar(downsample=10)
encoder_time, encoder_count = load_encoder(downsample=1)
lidar_length = len(lidar_time)  # (23173)
encoder_length = len(encoder_time)  # (58024)

fog_yaw = np.load('fog_yaw_10.npy')
MAP = map_init()

model = motion_obs_model()
n_particles = 1
x_part, y_part, theta_part, weight = model.init_particles(n_particles)

angles_range = np.linspace(-5, 185, 286) / 180 * np.pi

encoder_index = 0
lidar_index = 0
trajectory = np.array([[0], [0]])

for i in tqdm(range(1,len(encoder_time))):
    # use encoders and FOG data to compute instantaneous linear and angular velocities vt and wt
        tau = (encoder_time[encoder_index] - encoder_time[encoder_index - 1])/10**9 #get time difference in seconds
        if encoder_index == 0:
            count_l = encoder_count[encoder_index, 0]
            count_r = encoder_count[encoder_index, 1]
            lin_velocity = get_velocity(tau, count_l, count_r)
        else:
            count_l = encoder_count[encoder_index, 0] - encoder_count[encoder_index - 1, 0]
            count_r = encoder_count[encoder_index, 1] - encoder_count[encoder_index - 1, 1]
            lin_velocity = get_velocity(tau, count_l, count_r)
        # predict vehicle particle pose
        x_part, y_part, theta_part = model.predict_pose(x_part, y_part, theta_part, lin_velocity,
                                                        fog_yaw[encoder_index], tau,ADD_NOISE=False)
        trajectory = np.hstack((trajectory, np.array([x_part, y_part]).reshape(2, 1)))
        if encoder_index < encoder_length - 1:
            encoder_index += 1

plt.plot(trajectory[0],trajectory[1],color = "red")
plt.title("Final Trajectory")
plt.savefig("finaltrajec.png")
plt.show()
plt.pause(10)
