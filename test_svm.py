import os

from skimage.io import imsave
from sklearn.externals import joblib
import numpy as np

from constants import *
from segment_image import segment_slic

RGB_FROM_YUV = np.array([[1, 0, 1.13983],
                         [1, -0.39465, -.58060],
                         [1, 2.03211, 0]]).T
YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T
U_MAX = 0.436
V_MAX = 0.615

def clamp(val, low, high):
    return max(min(val, high), low)

def clampU(val):
    return clamp(val, -U_MAX, U_MAX)

def clampV(val):
    return clamp(val, -V_MAX, V_MAX)

def retrieveRGB(img):
    rgb = np.dot(img, RGB_FROM_YUV)
    for (i, j, k), value in np.ndenumerate(rgb):
        rgb[i][j][k] = clamp(rgb[i][j][k], 0, 1)
    return rgb

def retrieveYUV(img):
    return np.dot(img, YUV_FROM_RGB)

# Generates the adjacency list for each of the segments in the image.
def generate_adjacencies(segments, n_segments, img, subsquares):
    adjacency_list = []
    for i in range(n_segments):
        adjacency_list.append(set())
    for (i, j), value in np.ndenumerate(segments):
        # Check vertical adjacency
        if i < img.shape[0] - 1:
            newValue = segments[i + 1][j]
            if value != newValue and np.linalg.norm(subsquares[value] - subsquares[newValue]) < THRESHOLD:
                adjacency_list[value].add(newValue)
                adjacency_list[newValue].add(value)

        # Check horizontal adjacency
        if j < img.shape[1] - 1:
            newValue = segments[i][j + 1]
            if value != newValue and np.linalg.norm(subsquares[value] - subsquares[newValue]) < THRESHOLD:
                adjacency_list[value].add(newValue)
                adjacency_list[newValue].add(value)

    return adjacency_list

# Given the prior observed_u and observed_v, which are generated using the SVR,
# represent the system as a Markov Random Field and optimize over it using
# Iterated Conditional Modes. Return the prediction of the hidden U and V values
# of the segments.
# For now, we assume that the U and V channels behave independently.
def apply_mrf(observed_u, observed_v, segments, n_segments, img, subsquares):
    hidden_u = np.copy(observed_u)  # Initialize hidden U and V to the observed
    hidden_v = np.copy(observed_v)

    adjacency_list = generate_adjacencies(segments, n_segments, img, subsquares)

    for iteration in range(ICM_ITERATIONS):
        new_u = np.zeros(n_segments)
        new_v = np.zeros(n_segments)

        for k in range(n_segments):

            u_potential = 100000
            v_potential = 100000
            u_min = -1
            v_min = -1

            # Compute conditional probability over all possibilities of U
            for u in np.arange(-U_MAX, U_MAX, .001):
                u_computed = (u - observed_u[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    u_computed += WEIGHT_DIFF * ((u - hidden_u[adjacency]) ** 2)
                if u_computed < u_potential:
                    u_potential = u_computed
                    u_min = u
            new_u[k] = u_min

            # Compute conditional probability over all possibilities of V
            for v in np.arange(-V_MAX, V_MAX, .001):
                v_computed = (v - observed_v[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    v_computed += WEIGHT_DIFF * ((v - hidden_v[adjacency]) ** 2)
                if v_computed < v_potential:
                    v_potential = v_computed
                    v_min = v
            new_v[k] = v_min

        u_diff = np.linalg.norm(hidden_u - new_u)
        v_diff = np.linalg.norm(hidden_v - new_v)
        hidden_u = new_u
        hidden_v = new_v
        if u_diff < ITER_EPSILON and v_diff < ITER_EPSILON:
            break

    return hidden_u, hidden_v

# Given an image, predict its chrominance (U and V values in YUV space)


def predict_image(u_svr, v_svr, path, file):

    img, segments = segment_slic(path + file)
    yuv = retrieveYUV(img)   # Use first component of yuv to obtain black and white
    n_segments = segments.max() + 1

    # Construct the centroids of the image
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    luminance = np.zeros(n_segments)

    for (i, j), value in np.ndenumerate(segments):
        point_count[value] += 1
        centroids[value][0] += i
        centroids[value][1] += j
        luminance[value] += yuv[i][j][0]

    for k in range(n_segments):
        centroids[k] /= point_count[k]
        luminance[k] /= point_count[k]

    # Generate the subsquares
    subsquares = np.zeros((n_segments, SQUARE_SIZE * SQUARE_SIZE))
    for k in range(n_segments):
        # Check that the square lies completely within the image
        top = max(int(centroids[k][0]), 0)
        if top + SQUARE_SIZE >= img.shape[0]:
            top = img.shape[0] - 1 - SQUARE_SIZE
        left = max(int(centroids[k][1]), 0)
        if left + SQUARE_SIZE >= img.shape[1]:
            left = img.shape[1] - 1 - SQUARE_SIZE
        for i in range(0, SQUARE_SIZE):
            for j in range(0, SQUARE_SIZE):
                subsquares[k][i*SQUARE_SIZE + j] = yuv[i + top][j + left][0]
        subsquares[k] = np.fft.fft2(subsquares[k].reshape(SQUARE_SIZE, SQUARE_SIZE)).reshape(SQUARE_SIZE * SQUARE_SIZE)

    # Predict using SVR
    predicted_u = np.zeros(n_segments)
    predicted_v = np.zeros(n_segments)
    '''for k in range(n_segments):
        predicted_u[k] = clampU(u_svr.predict(subsquares[k])*2)
        predicted_v[k] = clampV(v_svr.predict(subsquares[k])*2)'''

    predicted_u = u_svr.predict(subsquares)
    predicted_v = v_svr.predict(subsquares)
    for k in range(n_segments):
        predicted_u[k] = clampU(predicted_u[k]*2)
        predicted_v[k] = clampV(predicted_v[k]*2)

    # Apply MRF to smooth out colorings
    predicted_u, predicted_v = apply_mrf(predicted_u, predicted_v, segments, n_segments, img, subsquares)

    # Reconstruct images
    for (i,j), value in np.ndenumerate(segments):
        yuv[i][j][1] = predicted_u[value]
        yuv[i][j][2] = predicted_v[value]
    rgb = retrieveRGB(yuv)

    # Compute the norm error. Note that it will be wildly incorrect if the img is b/w.
    error = 1000 * np.linalg.norm(rgb - img) / (img.shape[0] * img.shape[1])

    imsave('output/predicted/'+file, rgb)
    imsave('output/grayscale/'+file, yuv[:, :, 0])
    imsave('output/segment/'+file, segments)

    return error


if __name__ == '__main__':

    total_error = 0
    num_files = 0

    path = 'data/landscape-test/'
    files = os.listdir(path)

    u_svr = joblib.load('models/landscape-u-svr.model')
    v_svr = joblib.load('models/landscape-v-svr.model')

    print("Loaded the models")

    for file in files:
        total_error += predict_image(u_svr, v_svr, path, file)
        num_files += 1

    print('Total error for model', ':', total_error / num_files)