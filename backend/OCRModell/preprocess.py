import numpy as np
import cv2
import math
import pybobyqa

from collections import namedtuple
from typing import Tuple
import matplotlib.pyplot as plt

DeslantRes = namedtuple('DeslantRes', 'img, shear_val, candidates')
Candidate = namedtuple('Candidate', 'shear_val, score')

def _get_shear_vals(lower_bound: float,
                    upper_bound: float,
                    step: float) -> Tuple[float]:
    return tuple(np.arange(lower_bound, upper_bound + step, step))


def _shear_img(img: np.ndarray,
               s: float, bg_color: int,
               interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = img.shape
    offset = h * s
    w = w + int(abs(offset))
    tx = max(-offset, 0)

    shear_transform = np.asarray([[1, s, tx], [0, 1, 0]], dtype=float)
    img_sheared = cv2.warpAffine(img, shear_transform, (w, h), flags=interpolation, borderValue=bg_color)

    return img_sheared

def _compute_score(img_binary: np.ndarray, s: float) -> float:
    img_sheared = _shear_img(img_binary, s, 0)
    h = img_sheared.shape[0]

    img_sheared_mask = img_sheared > 0
    first_fg_px = np.argmax(img_sheared_mask, axis=0)
    last_fg_px = h - np.argmax(img_sheared_mask[::-1], axis=0)
    num_fg_px = np.sum(img_sheared_mask, axis=0)

    dist_fg_px = last_fg_px - first_fg_px
    col_mask = np.bitwise_and(num_fg_px > 0, dist_fg_px == num_fg_px)
    masked_dist_fg_px = dist_fg_px[col_mask]

    score = sum(masked_dist_fg_px ** 2)
    return score

def deslant(img: np.ndarray, lower_bound: float = -2, upper_bound: float = 2, num_steps: int = 20, bg_color=255) -> DeslantRes:
    img_binary = cv2.threshold(255 - img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] // 255
    step = (upper_bound - lower_bound) / num_steps
    shear_vals = _get_shear_vals(lower_bound, upper_bound, step)
    candidates = [Candidate(s, _compute_score(img_binary, s)) for s in shear_vals]
    best_shear_val = sorted(candidates, key=lambda c: c.score, reverse=True)[0].shear_val

    res_img = _shear_img(img, best_shear_val, bg_color, cv2.INTER_LINEAR)
    return res_img

def createKernel(kernelSize, sigma, theta):
    assert kernelSize % 2
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def segment(imagenp):
    img = cv2.imdecode(imagenp, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernelSize=25
    sigma=11
    theta=7

    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -5, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)

    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    imgThres = np.float32(imgThres)

    hist = cv2.reduce(imgThres,1, cv2.REDUCE_AVG).reshape(-1)

#   Hisztogram megtekintése:
#    plt.plot(hist)
#    plt.xlabel('Sor szám')
#    plt.ylabel('Pixelek átlagolt száma')
#    plt.show()

    th = 2
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    lines = []
    sumHeight = 0

    for y in range(len(uppers)):
        lines.append(img[uppers[y]:lowers[y], 0:W])
        sumHeight += lowers[y]-uppers[y]

    avarage = sumHeight / len(uppers)

    for x in lines:
        if x.shape[0] < avarage*0.5:
            lines.remove(x)

    response = []
    for x in lines:
        response.append(deslant(x))

    return response
