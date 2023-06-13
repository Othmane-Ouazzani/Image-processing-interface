import cv2
import numpy as np

def rotateImg(img, angle):#Done
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

def binarizeThreshold(img, threshold):#Done
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = img.copy()
    bin_img[bin_img < threshold] = 0 
    bin_img[bin_img >= threshold] = 255
    return bin_img

def binarizeOtsu(img):#Done
    histog = cv2.calcHist([img], [0], None, [256], [0, 256])
    histog = histog.flatten()
    weightB = np.zeros(256)
    weightF = np.zeros(256)
    varB = np.zeros(256)
    varF = np.zeros(256)
    for i in range(256):
        weightB[i] = np.sum(histog[0:i+1])/np.sum(histog)
        weightF[i] = np.sum(histog[i+1:])/np.sum(histog)
        meanB = 0
        meanF = 0
        if np.sum(histog[0:i+1]) == 0:
            meanB = 0
        else:
            for l in range(i+1):
                meanB += l * histog[l]
            meanB /= np.sum(histog[0:i+1])
        if np.sum(histog[i+1:]) == 0:
            meanF = 0
        else:
            for k in range(i+1, 256):
                meanF += k * histog[k]
            meanF /= np.sum(histog[i+1:])
        for m in range(i+1):
            varB[i] += ((m-meanB)**2) * histog[m]
        for n in range(i+1, 256):
            varF[i] += ((n-meanF)**2) * histog[n]
        if np.sum(histog[0:i+1]) == 0:
            varB[i] = 0
        else:
            varB[i] /= np.sum(histog[0:i+1])
        if np.sum(histog[i+1:]) == 0:
            varF[i] = 0
        else:
            varF[i] /= np.sum(histog[i+1:])
    withinClassVar = np.zeros(256)
    for j in range(256):
        withinClassVar[j] = weightB[j] * varB[j] + weightF[j] * varF[j]
    thresholding = np.argmin(withinClassVar)
    imgBin = cv2.threshold(img, thresholding, 255, cv2.THRESH_BINARY)[1]
    
    return imgBin

def getHistogram(img):#Done
    hist = np.zeros((256,), dtype=np.int32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = img[i, j]
            hist[intensity] += 1
    return hist

def equalization(img):#Done
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    img_eq = cv2.LUT(img, cdf_normalized.astype(np.uint8))

    return img_eq

def filt_moy(I, k):
    offset = k // 2
    img = np.double(I) / np.double(np.max(I))

    img_padded = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_REPLICATE)

    img_filtered = np.zeros_like(img)

    for i in range(offset, img_padded.shape[0] - 2 * offset):
        for j in range(offset, img_padded.shape[1] - 2 * offset):
            img_filtered[i - offset, j - offset] = np.mean(img_padded[i:i+2*offset+1,j:j+2*offset+1])

    img_filtered = img_filtered[offset:-offset, offset:-offset]
    img_filtered = np.uint8(img_filtered * np.double(np.max(I)))

    return img_filtered

def filt_med(I, k):
    offset = k // 2
    img = np.double(I) / np.double(np.max(I))

    img_padded = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_REPLICATE)

    img_filtered = np.zeros_like(img)

    for i in range(offset, img_padded.shape[0] - 2 * offset):
        for j in range(offset, img_padded.shape[1] - 2 * offset):
            T = img_padded[i:i+2*offset+1,j:j+2*offset+1]
            img_filtered[i - offset, j - offset] = np.median(T)

    img_filtered = img_filtered[offset:-offset, offset:-offset]
    img_filtered = np.uint8(img_filtered * np.double(np.max(I)))

    return img_filtered

# img = cv2.imread("images/lena.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# pimg = equalization(img)
# cv2.imshow("img",pimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # import matplotlib.pyplot as plt

# # plt.plot(pimg)
# # plt.show()