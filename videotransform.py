import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

vid = cv2.VideoCapture("dance.mp4")
count = 0

os.mkdir('resume')
os.mkdir('data')

while vid.isOpened():
    ret, frame = vid.read()

    if ret == True:
        print('Read %d frame: ' % count, ret)
        framepath = os.path.join('data', "frame{:d}.jpg".format(count))
        plotpath = os.path.join('resume', "frame{:d}.jpg".format(count))
        count += 1

        cv2.imwrite(framepath, frame)  # save frame as JPEG file
        
        # * process image
        plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

        img_original   = cv2.imread(framepath, 0)
        img_spectrum   = np.fft.fft2(img_original)
        img_center     = np.fft.fftshift(img_spectrum)
        img_inv_center = np.fft.ifftshift(img_center)
        img_processed  = np.fft.ifft2(img_inv_center)
        
        plt.subplot(151), plt.imshow(img_original, "gray"), plt.title("Original Image")
        plt.subplot(152), plt.imshow(np.log(1+np.abs(img_spectrum)), "gray"), plt.title("Spectrum")
        plt.subplot(153), plt.imshow(np.log(1+np.abs(img_center)), "gray"), plt.title("Centered Spectrum")
        plt.subplot(154), plt.imshow(np.log(1+np.abs(img_inv_center)), "gray"), plt.title("Decentralized")
        plt.subplot(155), plt.imshow(np.abs(img_processed), "gray"), plt.title("Processed Image")

        os.remove(framepath)
        
        plt.savefig(plotpath, bbox_inches='tight')
    else:
        break

vid.release()
cv2.destroyAllWindows()

