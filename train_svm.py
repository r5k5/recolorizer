import os

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import numpy as np
from constants import *
from segment_image import segment_slic

YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T


def retrieveYUV(img):
    return np.dot(img, YUV_FROM_RGB)


def generateSubsquares(path):
    img, segments = segment_slic(path)
    yuv = retrieveYUV(img)
    n_segments = segments.max() + 1

    # Compute the centroids/average U and V of each of the superpixels
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    U = np.zeros(n_segments)
    V = np.zeros(n_segments)
    for (i, j), value in np.ndenumerate(segments):
        point_count[value] += 1
        centroids[value][0] += i
        centroids[value][1] += j
        U[value] += yuv[i][j][1]
        V[value] += yuv[i][j][2]

    for k in range(n_segments):
        centroids[k] /= point_count[k]
        U[k] /= point_count[k]
        V[k] /= point_count[k]

    # Generate the array of squares
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

    return subsquares, U, V


if __name__ == '__main__':

    X = np.array([]).reshape(0, SQUARE_SIZE * SQUARE_SIZE)
    U_L = np.array([])
    V_L = np.array([])

    path = 'data/landscape-train/'
    files = os.listdir(path)

    for file in files:

        print("Training on", file)
        subsquares, U, V = generateSubsquares(path+file)

        X = np.concatenate((X, subsquares), axis=0)
        U_L = np.concatenate((U_L, U), axis=0)
        V_L = np.concatenate((V_L, V), axis=0)

    u_svr = SVR(C=C, epsilon=SVR_EPSILON)
    v_svr = SVR(C=C, epsilon=SVR_EPSILON)
    
    print('Fitting the model given by C =', C, ', epsilon =', SVR_EPSILON, 'for U channel')
    u_svr.fit(X, U_L)
    print('Fitting the model given by C =', C, ', epsilon =', SVR_EPSILON, 'for V channel')
    v_svr.fit(X, V_L)
    joblib.dump(u_svr, 'models/landscape-u-svr.model')
    joblib.dump(v_svr, 'models/landscape-v-svr.model')

    '''u_nn = MLPRegressor(hidden_layer_sizes=(5, 2), alpha=0.00001)
    v_nn = MLPRegressor(hidden_layer_sizes=(5, 2), alpha=0.00001)
    print('Fitting the model given by for U channel')
    u_nn.fit(X, U_L)
    print('Fitting the model given by for V channel')
    u_nn.fit(X, V_L)
    joblib.dump(u_nn, 'models/image-net-200-u-nn.model')
    joblib.dump(u_nn, 'models/image-net-200-v-nn.model')'''
