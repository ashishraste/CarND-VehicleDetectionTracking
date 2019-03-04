from tracking_util import *
import numpy as np
import matplotlib.pyplot as plt
from vehicle_classifier import FeatureParams
import glob
import pickle
from scipy.ndimage.measurements import label
from pathlib import Path

image_data = load_dataset()
vehicles = image_data['vehicles']
non_vehicles = image_data['non_vehicles']

model_dir = Path("../")
test_images_dir = Path("test_images")
output_images_dir = Path("output_images")

with open(str(model_dir/'model.p'), 'rb') as file:
    model = pickle.load(file)
svc = model['classifier']
scaler = model['scaler']


def display_images(img1, img2, img1_title=None, img2_title=None, gray_cmap=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if img1_title is not None and img2_title is not None:
        ax1.set_title(img1_title, fontsize=30)
        ax2.set_title(img2_title, fontsize=30)
    ax1.axis('off')
    ax2.axis('off')
    if gray_cmap:
        ax1.imshow(img1, cmap='gray')
        ax2.imshow(img2, cmap='gray')
    else:
        ax1.imshow(img1)
        ax2.imshow(img2)
    plt.show()


def display_hog_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    params = FeatureParams()
    _,hog_img = get_hog_features(img_gray, orient=params.orient, pix_per_cell=params.pix_per_cell, cell_per_block=params.cell_per_block,
                     vis=True, feature_vec=False)
    display_images(img, hog_img, 'Source Image', 'HOG image '+params.color_space, gray_cmap=True)


def display_car_detections():
    test_images = list(map(lambda img_path: load_image(img_path), glob.glob(str(test_images_dir/'*.jpg'))))
    detected_car_images = list(
        map(
            lambda test_img: draw_boxes(
                test_img, search_car_windows(
                    test_img, svc, FeatureParams(), scaler)), test_images))

    for i in range(len(test_images)):
        display_images(test_images[i], detected_car_images[i])
        save_image(detected_car_images[i], str(output_images_dir/'car_detections'+str(i)+'.png'))


def display_car_heatmap(img):
    detected_windows = search_car_windows(img, svc, FeatureParams(), scaler)
    # Add heat to each box in box list
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, detected_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)

    print(labels[1], 'cars found')
    plt.axis('off')
    plt.imshow(labels[0], cmap='gray')
    plt.show()
    save_image(labels[0], str(output_images_dir/'car_heatmap6.png'), cmap_gray=True)
    return heatmap


def display_car_bboxes(img, heatmap):
    labels = label(heatmap)
    bbox_img = draw_labeled_bboxes(img, labels)
    plt.axis('off')
    plt.imshow(bbox_img)
    plt.show()
    save_image(bbox_img, str(output_images_dir/'car_bboxes6.png'))



if __name__ == '__main__':
    # print('Number of vehicle-images: {}'.format(len(vehicles)))
    # print('Number of non-vehicle-images: {}'.format(len(non_vehicles)))
    rand_idx1 = np.random.randint(0, len(vehicles))
    rand_idx2 = np.random.randint(0, len(non_vehicles))
    vehicle_img = vehicles[rand_idx1]
    non_vehicle_img = non_vehicles[rand_idx2]

    # Display sample vehicle and non-vehicle images from the dataset.
    display_images(vehicle_img, non_vehicle_img, 'Sample vehicle image', 'Sample non-vehicle image')

    # Display HOG features.
    display_hog_features(vehicle_img)
    display_hog_features(non_vehicle_img)

    # Display car detections.
    display_car_detections()

    # Display detected car images' heatmap.
    test_img = load_image(str(test_images_dir/'test6.jpg'))
    heatmap = display_car_heatmap(test_img)
    display_car_bboxes(test_img, heatmap)
