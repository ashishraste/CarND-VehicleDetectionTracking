import numpy as np
import pickle
import glob
from tracking_util import load_image

'''
Loads an image dataset and saves (pickles) them as numpy arrays.
This script pickles the training images containing vehicles and non-vehicles.
'''

vehicles = []
non_vehicles = []

vehicle_images = glob.glob('../training_images/vehicles/*.png')
non_vehicle_images = glob.glob('../training_images/non-vehicles/*.png')


for v_img in vehicle_images:
    vehicles.append(load_image(v_img))

for nv_img in non_vehicle_images:
    non_vehicles.append(load_image(nv_img))


image_dict = {'vehicles': np.array(vehicles), 'non_vehicles': np.array(non_vehicles)}
with open('../image_data.p', 'wb') as file:
    pickle.dump(image_dict, file)