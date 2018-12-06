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

        original_color = cv2.imread(dir_name + '/' + file)
        h, w, c = original_color.shape
        original = cv2.imread(dir_name + '/' + file, 0)
        rt, binary = cv2.threshold(original, 110, 255, cv2.THRESH_BINARY_INV)
        med = cv2.medianBlur(binary, 5)

        kernel = np.ones((11, 11), np.uint8)

        circles = cv2.HoughCircles(med, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=int(min(h,w) / 4), maxRadius=min(h, w))


        if circles is not None:

            #ori = cv2.cvtColor(original_color,cv2.COLOR_BGR2RGB)
            #plt.imshow(ori)
            #plt.show()

            original_final = cv2.imread(dir_name + '/' + file)
            circles = np.uint16(np.around(circles))
            x = 0
            y = 0
            r = 0
            cnt = 0
            for i in circles[0, :]:
                x += i[0]
                y += i[1]
                r += i[2]
                cnt += 1
                cv2.circle(original_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(original_color, (i[0], i[1]), 2, (0, 0, 255))

            x = int(x / cnt)
            y = int(y / cnt)
            r = int(r / cnt)

            cv2.circle(original_color, (x, y), r, (0, 0, 255), 5)
            # plt.imshow(original_color)
            #plt.show()

            original_final_copy = original_final.copy()
            h, w, c = original_final_copy.shape
            min_y = max((y - r), 0)
            max_y = min((y + r), h)
            min_x = max((x - r), 0)
            max_x = min((x + r), w)
            cropped_image = original_final_copy[min_y: max_y, min_x: max_x]

            hc, wc, cc = cropped_image.shape
            for ix in range(hc):
                for j in range(wc):
                    d = math.sqrt((ix - r) * (ix - r) + (j - r) * (j - r))
                    if d >= r or (cropped_image[ix, j, 0] < 20 and cropped_image[ix, j, 1] < 20 and cropped_image[ix, j, 2] < 20):
                        cropped_image[ix, j] = [0, 0, 0]

            cv2.circle(original_final, (x, y), r, (0, 0, 255), 5)

            blue = np.average(cropped_image[:, :, 0])
            green = np.average(cropped_image[:, :, 1])
            red = np.average(cropped_image[:, :, 2])
            (means, stds) = cv2.meanStdDev(cropped_image)
            list.append([file, stds[0], stds[1], stds[2]])
            #print(red,blue,green)
            if red < blue or red < green:
                print("red<blue")
                #continue
            if red < 150.00:
                print("red<150")
                #continue


            if stds[0,0] > 40:
                print("stds>40")
                #continue


            #ori = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            #plt.imshow(ori)
            #plt.show()
            opt_img = cropped_image.copy()
            gray_img = cv2.cvtColor(opt_img, cv2.COLOR_BGR2GRAY)
            maxi_pixel = np.max(gray_img)
            ret, th = cv2.threshold(gray_img, maxi_pixel - maxi_pixel * 0.02, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            hi_mask = cv2.dilate(th, kernel, iterations=3)
            specular = cv2.inpaint(cropped_image, hi_mask, 2, flags=cv2.INPAINT_TELEA)

            #ori = cv2.cvtColor(specular, cv2.COLOR_BGR2RGB)
            #plt.imshow(ori)
            #plt.show()

            rgb_img = specular.copy()
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

            rt, binary = cv2.threshold(temp, 160, 255, cv2.THRESH_BINARY)
            avg = np.average(binary[:,:])
            print(avg)
            if (avg < 2) :
                continue
            print(file)

            cv2.imwrite('o1' + '/' + 'cropped_' + file, cropped_image)
            cv2.imwrite('e1' + '/' + 'cropped_enhanced_' + file, specular)
            cv2.imwrite('v1' + '/' + 'cropped_enhanced_vein' + file, temp)


with open('result.txt', 'w') as f:
    f.write('filename,x,y,r\n')
    for res in list:
        f.write(res[0] + ',' + str(res[1]) + ',' + str(res[2]) + ',' + str(res[3]) + '\n')
f.close()

print()
print()
print('result stored in file====>result.txt')
print()
