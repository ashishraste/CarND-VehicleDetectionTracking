from tracking_util import *
from moviepy.editor import VideoFileClip
from vehicle_classifier import FeatureParams
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from pathlib import Path

model_dir = Path("../")
test_images_dir = Path("../test_images")
output_dir = Path("../")

class HotWindows():
    def __init__(self, num_frames=30):
        self.windows_q = []
        self.num_frames = num_frames

    def add_windows(self, windows):
        self.windows_q.append(windows)
        if len(self.windows_q) > self.num_frames:
            self.windows_q.pop(0)

    def get_windows(self):
        all_windows = []
        for windows in self.windows_q:
            all_windows += windows
        return all_windows


# Globals.
hot_windows = HotWindows()
feature_params = FeatureParams()
svc = None  # Linear SVM classifier gets loaded when running the video-pipeline.
scaler = None  # Feature scaler.
heatmap_thresh = 33


def vehicle_detection_pipeline(src_img):
    global svc, scaler, hot_windows, feature_params
    detected_windows = search_car_windows(src_img, svc, FeatureParams(), scaler)

    # Add new vehicle-detections to list of existing detections.
    hot_windows.add_windows(detected_windows)
    all_hot_windows = hot_windows.get_windows()

    # Draw heatmap over detected vehicle windows.
    heat = np.zeros((720,1200))
    heat = apply_threshold(add_heat(heat, all_hot_windows), heatmap_thresh)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)

    # Draw final bounding boxes
    out_img = draw_labeled_bboxes(np.copy(src_img), labels)
    return out_img


def run_video_pipeline(input_video_file, output_video_file):
    global svc, scaler
    with open(str(model_dir/'model.p'), 'rb') as file:
        model = pickle.load(file)
    svc = model['classifier']
    scaler = model['scaler']

    input = VideoFileClip(input_video_file)
    detected_vehicle_video = input.fl_image(vehicle_detection_pipeline)
    detected_vehicle_video.write_videofile(output_video_file, audio=False)


if __name__ == '__main__':
    run_video_pipeline(str(output_dir/'project_video.mp4'), str(output_dir/'detected_vehicles.mp4'))

    # with open(str(model_dir/'model.p'), 'rb') as file:
    #     model = pickle.load(file)
    # svc = model['classifier']
    # scaler = model['scaler']
    # vehicle_detection_pipeline(load_image(str(test_images_dir/'test6.jpg')))