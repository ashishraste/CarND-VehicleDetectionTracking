from tracking_util import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from pathlib import Path

# Load positive and negative features.
images = load_dataset()
cars = images['vehicles']
notcars = images['non_vehicles']

model_dir = Path("../")

class FeatureParams():
    """
    Holds features' parameters.
    """
    def __init__(self):
        # Spatial parameters.
        self.spatial_size = (32,32)
        # Color selection parameters.
        self.hist_bins = 32
        # HOG parameters.
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 8
        self.pix_per_cell = 16
        self.cell_per_block = 4
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"


def train():
    # Selected features.
    spatial_feat = True
    hist_feat = True
    hog_feat = True
    params = FeatureParams()

    # Extract features from positive and negative datasets.
    positive_features = extract_features(cars, params, spatial_feat, hist_feat, hog_feat)
    negative_features = extract_features(notcars, params, spatial_feat, hist_feat, hog_feat)

    # Create an array stack of feature vectors.
    X = np.vstack((positive_features, negative_features, negative_features, negative_features)).astype(np.float64)

    # Define the labels vector.
    y = np.hstack((np.ones(len(positive_features)), np.zeros(3*len(negative_features))))

    # Split up data into randomized training and test sets.
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state, shuffle=True)

    # Fit a per-column scaler only on the training data.
    X_scaler = StandardScaler().fit(X_train)

    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC(verbose=1, max_iter=3000)
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample.
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    # Save the classifier model.
    model = {'classifier': svc, 'scaler': X_scaler}
    pickle.dump(model, open(str(model_dir/'model.p'), 'wb'))


if __name__ == '__main__':
    train()