# Based on https://github.com/ducha-aiki/local_feature_tutorial repo with some modification.

__all__ = ['visualize_grid', 'decolorize', 'draw_matches_cv2', 'draw_matches']

# Cell
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
from typing import List
def visualize_grid(img_fnames:List[str], figsize=(16,16)):
    num_imgs = len(img_fnames)
    cols = int(np.sqrt(num_imgs))
    if num_imgs % cols == 0:
        rows = (num_imgs // cols)
    else:
        rows = (num_imgs // cols) +1
    fig = plt.figure(1, figsize)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(rows, cols),
                     axes_pad=0)
    for i in range(num_imgs):
        grid[i].imshow(cv2.cvtColor(cv2.imread(img_fnames[i]), cv2.COLOR_BGR2RGB))
        grid[i].axis('off')
    return


# Cell
from copy import deepcopy
def decolorize(img):
    return  cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
def draw_matches_cv2(kps1, kps2, img1, img2, figsize=(12,8), mask = None):
    if type(img1) is str:
        img1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    if type(img2) is str:
        img2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
    h,w,ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    if mask is None:
        mask = [True for i in range(len(kps1))]
    # Blue is estimated, green is ground truth homography
    draw_params = dict( matchColor = (0,255,0), # draw matches in yellow color
                   singlePointColor = (0,255,0),
                   matchesMask = mask, # draw only inliers
                   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_out = cv2.drawMatches(img1,kps1,img2,kps2,
                              [cv2.DMatch(i,i, 0) for i in range(len(kps1))],None,**draw_params)
    plt.figure(figsize=figsize)
    plt.imshow(img_out)
    return

# Cell
import matplotlib.pyplot as plt
def draw_matches(kp1, kp2, img1, img2, path,  color=None,  figsize=(12,8), mask = None, vert = False, R=1):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    if type(img1) is str:
        img1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    if type(img2) is str:
        img2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if not vert:
        if len(img1.shape) == 3:
            new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    else:
        if len(img1.shape) == 3:
            new_shape = (img1.shape[0]+ img2.shape[0], max(img1.shape[1],img2.shape[1]), img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (img1.shape[0]+ img2.shape[0], max(img1.shape[1],img2.shape[1]))
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    if not vert:
        new_img[0:img1.shape[0],0:img1.shape[1]] = img1
        new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    else:
        new_img[0:img1.shape[0],0:img1.shape[1]] = img1
        new_img[img1.shape[0]:img1.shape[0]+img2.shape[0], 0:img2.shape[1]] = img2
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 8
    if color is not None:
        c = color
    matches = [cv2.DMatch(i,i, 0) for i in range(len(kp1))]


    def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return s<m

    arr = np.zeros(len(kp1))
    for kk in range(len(kp1)):
      M = (kp1[kk].pt[0] - kp2[kk].pt[0]) / (kp1[kk].pt[1] - kp2[kk].pt[1] + 1e-5)
      arr[kk] = M
      # print(M)
    arr_logic = reject_outliers(arr, m=40)
    arr = arr[arr_logic]

    title = "All matches: {:d}" \
          .format(len(kp1))

    for idx, m in enumerate(matches):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if color is None:
            c = np.random.randint(0,256,3).tolist() if len(img1.shape) == 3 else np.random.randint(0,256)
        else:
            c = color
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        if not vert:
            end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        else:
            end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([0, img1.shape[0]]))

        cv2.line(new_img, end1, end2, c, thickness)

    plt.figure(figsize=figsize)
    plt.imshow(new_img)
    plt.title(title)
    plt.axis('off')
    plt.savefig('%s.JPG'%path,bbox_inches='tight', dpi=100, quality=90)
    plt.close()
    return 
