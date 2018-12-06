import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy import signal

cwd = os.getcwd()
image_dir = os.path.join(cwd, 't')
dir_name = 't'
list = []
for (_, _, files) in os.walk(image_dir):
    for file in files:
        print(file)

        original_color = cv2.imread(dir_name + '/' + file)
        blue = np.average(original_color[:, :, 0])
        green = np.average(original_color[:, :, 1])
        red = np.average(original_color[:, :, 2])

        print(red,blue,green)
        if red < blue and red < green:
            continue
        if red < 150.00:
            continue
        (means, stds) = cv2.meanStdDev(original_color)
        print(stds)
        if stds[0, 0] > 40:
            continue
        h, w, c = original_color.shape
        continue
        if True:
            rgb_img = original_color.copy()
            green_img = rgb_img.copy()
            green_img[:, :, 0] = 0
            green_img[:, :, 2] = 0

            Y_img = cv2.cvtColor(green_img, cv2.COLOR_RGB2YCrCb)
            Y_img[:, :, 2] = 0
            Y_img[:, :, 1] = 0
            Y_img = cv2.cvtColor(Y_img, cv2.COLOR_YCrCb2RGB)
            Y_img = cv2.cvtColor(Y_img, cv2.COLOR_RGB2GRAY)
            maxi = np.max(Y_img)
            mini = np.min(Y_img)

            for i in range(Y_img.shape[0]):
                for j in range(Y_img.shape[1]):
                    pixel = Y_img[i, j]
                    Y_img[i, j] = ((pixel - mini) / (maxi - mini)) * 255

            H = []
            H.append([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
            H.append([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
            H.append([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
            H.append([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
            H.append([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
            H.append([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
            H.append([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
            H.append([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])

            H = np.array(H)
            gradient = []

            for i in range(8):
                grad = signal.convolve2d(Y_img, H[i], fillvalue=1)
                gradient.append(grad)

            gradient = np.array(gradient)
            final_grad = []

            for i in range(Y_img.shape[0]):
                a = []
                for j in range(Y_img.shape[1]):
                    maxi = -9999
                    for p in range(8):
                        if gradient[p][i][j] > maxi:
                            maxi = gradient[p][i][j]
                    a.append(maxi)
                final_grad.append(a)

            final_grad = np.array(final_grad)

            temp = Y_img.copy()

            for i in range(Y_img.shape[0]):
                for j in range(Y_img.shape[1]):
                    if (final_grad[i][j] < 80):
                        temp[i, j] = 0


            #ori = cv2.cvtColor(temp,cv2.COLOR_GRAY2RGB)
            #plt.imshow(ori)
            #plt.show()
            (means, stds) = cv2.meanStdDev(temp)
            print(stds[0][0],means[0][0])

print()
print()
print('result stored in file====>result.txt')
print()
