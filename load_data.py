from pr2_utils import *
import os

def load_lidar(downsample = 1):
    """
    downsample : Downsample the dataset by 'downsample' times
    This function returns lidar timestamps and ranges
    """
    ts, ranges = read_data_from_csv('data/sensor_data/lidar.csv')
    return ts[::downsample],ranges[::downsample]

def load_encoder(downsample = 1):
    """
    downsample : Downsample the dataset by 'downsample' times
    This function returns encoder timestamps and counts
    """
    ts, data = read_data_from_csv('data/sensor_data/encoder.csv')
    return ts[::downsample], data[::downsample]

def load_FOG(downsample = 1):
    """
    downsample : Downsample the dataset by 'downsample' times
    This function returns FOG timestamps and delta roll, pitch and yaw readings
    """
    ts, data = read_data_from_csv('data/sensor_data/fog.csv')
    return ts[::downsample], data[::downsample]

def load_stereo():
    """
    This function returns timestamps of left stereo images
    """
    stereo_time = os.listdir('data/stereo_images/stereo_left')
    return np.array(stereo_time).astype('<U19')

