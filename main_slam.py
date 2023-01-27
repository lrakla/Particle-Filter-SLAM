import matplotlib.pyplot as plt

from MAP import update_map, map_init, valid_range, xy2map
from load_data import load_lidar, load_encoder
from model import motion_obs_model, get_velocity
from transforms import *
from texture_map import *
import os

# Load all the data
lidar_time, lidar_scans = load_lidar(downsample=10)
encoder_time, encoder_count = load_encoder(downsample=1)
fog_yaw = np.load('fog_yaw_10.npy')

# TS of precomputed disparity images

# stereo_times = os.listdir('disparity')
# stereo_times = np.array(stereo_times).astype('<U19')
lidar_length = len(lidar_time)
encoder_length = len(encoder_time)
#stereo_length = len(stereo_times)

# synchronize encoder and FOG by summing every 10 fog readings
# This is done as FOG is sampled 10x encoder sample rate
# fog_yaw = pd.DataFrame(fog_yaw)
# fog_yaw_synced = fog_yaw.groupby(fog_yaw.index // 10).sum()
# np.save('fog_yaw_10.npy', fog_yaw_synced)


MAP = map_init()
text_map = np.zeros((MAP['sizex'], MAP['sizey'], 3))

# convert first lidar scan to valid range
angles_range = np.linspace(-5, 185, 286) / 180 * np.pi
lidar_scan, angles = valid_range(lidar_scans[0, :], angles_range)

# Initialize instance of model class
model = motion_obs_model()

# change this value to increase/decrease particles
n_particles = 1
# all values are initialized to zero, and weight is initialized to 1/N
x_part, y_part, theta_part, weight = model.init_particles(n_particles)

# Update MAP with the first lidar scan using 1st particle state
MAP = update_map(x_part[:, 0], y_part[:, 0], theta_part[:, 0], lidar_scan, angles, MAP)

# Initialize sensor indices
encoder_idx = 0
lidar_idx = 0
stereo_idx = 0

# Intiliaze trajectory as 0,0 as we assume vehicle starts as 0,0
trajectory = np.array([[0], [0]])

predictions = 0
updations = 0
iterations = len(encoder_time) + len(lidar_time)

# BEGIN SLAM
print("-----------SLAM BEGINS---------------")
for i in tqdm(range(iterations)):
    if i % 10000 == 0 or i == iterations - 1:
        # Converting log odds MAP to 1 and 0 using PMF (using sigmoid function)
        img_map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.5).astype(np.int)
        #cell is occupied if less than 0.5 and coloured white. Hence image shows white for
        # occupied and black for free
        # convert end point in world frame to cells

        ex, ey = xy2map(trajectory[0, :], trajectory[1, :])

        indGood = np.logical_and(np.logical_and(np.logical_and((ex > 1), (ey > 1)), (ex < MAP['sizex'])),
                                 (ey < MAP['sizey']))

        img_map[ex[indGood], ey[indGood]] = 1
        # output_texture_map[ex[indGood], ey[indGood]] = np.array([255, 255, 0])

        # plot map
        # plt.imshow(MAP['map'], cmap='gray')
        # plt.title("Occupancy Map")
        # plt.pause(10)

        # save map
        if i % 10000 == 0 or i == iterations - 1:
        # if i == iterations - 1:
            plt.imshow(img_map, cmap="gray")
            plt.title(f"Occupancy Map at {i} iterations")
            #plt.savefig(str(i) + '-Occupancy map-np' + str(n_particles) + '.png')

    # -----------------------PREDICTION (when encoder reading occurs before lidar scan)---------------------------------------------

    if encoder_idx < encoder_length and lidar_idx < lidar_length and encoder_time[encoder_idx] < lidar_time[lidar_idx]:

        # use encoder data to compute instantaneous linear velocity
        tau = (encoder_time[encoder_idx] - encoder_time[encoder_idx - 1]) / 10 ** 9
        if encoder_idx == 0:
            lin_velocity = 0  # initialize linear velocity as 0
        else:
            count_l = encoder_count[encoder_idx, 0] - encoder_count[encoder_idx - 1, 0]
            count_r = encoder_count[encoder_idx, 1] - encoder_count[encoder_idx - 1, 1]
            lin_velocity = get_velocity(tau, count_l, count_r)
        # print('linear velocity:', lin_velocity)
        # Skip cycle of SLAM if car is not moving
        if encoder_idx != 0 and lin_velocity < 0.01:
            encoder_idx += 1
            continue

        # PREDICT pose of the vehicle using n_particles
        predictions += 1

        x_part, y_part, theta_part = model.predict_pose(x_part, y_part, theta_part, lin_velocity,
                                                        fog_yaw[encoder_idx], tau, ADD_NOISE=True)

        if encoder_idx < encoder_length - 1:
            encoder_idx += 1
        else:
            lidar_idx += 1

    # -----------------UPDATE (when lidar scan happens before encoder)-----------------------------

    elif lidar_idx < lidar_length:
        updations += 1

        # convert lidar scan to valid range
        lidar_scan, angles = valid_range(lidar_scans[lidar_idx, :], angles_range)

        # update particle weights using mapCorrelation
        updated_weight = model.update_pose(x_part, y_part, theta_part, weight, lidar_scan, angles,
                                           n_particles, MAP)

        # --------------MAPPING (update map with the pose of the max weighted particle) ----------------------------------------------------------------------
        max_idx = np.argmax(updated_weight)
        max_x_part, max_y_part, max_theta_part = x_part[:, max_idx], y_part[:, max_idx], theta_part[:, max_idx]
        MAP = update_map(max_x_part, max_y_part, max_theta_part, lidar_scan, angles, MAP)
        # Update trajectory by adding pose of max weighted particle
        trajectory = np.hstack((trajectory, np.array([max_x_part, max_y_part]).reshape(2, 1)))
        # if updations % 1000 == 0:
        #     plt.scatter(trajectory[0], trajectory[1], color = 'red')
        #     plt.title(f"Trajectory Map at {i} iterations")
        #     plt.savefig(f"Trajectory{i}.png")

        ## Texture Mapping
        # if lidar_idx % 100 == 0:
        #     stereo_idx += 1
        #     disparity = np.load("disparity/%s.npy" % stereo_times[stereo_idx])
        #     path_l = 'data/stereo_images/stereo_left/%s.png' % stereo_times[stereo_idx]
        #     image_l = cv2.imread(path_l, 0)
        #     svc = get_world_coords_from_uv(disparity)
        #     text_map = texture_map(MAP, max_x_part, max_y_part, max_theta_part, svc, image_l,
        #                            text_map)
        #plot map
        # plt.imshow(text_map, cmap='gray')
        # plt.title("Texture Map")
        # plt.pause(10)

        # -------------------------RESAMPLING-------------------------------------------

        N_eff = 1 / np.dot(weight.reshape(1, n_particles), weight.reshape(n_particles, 1))
        if N_eff < 0.6 * n_particles:
            x_part, y_part, theta_part, weight = model.resampling(x_part, y_part, theta_part, weight, n_particles)

        if lidar_idx < lidar_length:
            lidar_idx += 1
        else:
            encoder_idx += 1

print(f"Final Predictions {predictions}, Final Updations {updations}")
plt.imshow(MAP['map'], cmap='gray')
plt.title("Occupancy Map")
plt.pause(20)
if not exists('final_trajectory.npy'):
    np.save('final_trajectory.npy', trajectory)
else:
    trajectory = np.load('final_trajectory.npy')

plt.scatter(trajectory[0], trajectory[1])
plt.title("Trajectory Map")
plt.pause(10)
#plt.imsave('Finished Occupancy map.png', MAP['map'], cmap='gray')
