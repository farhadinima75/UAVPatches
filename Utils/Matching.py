# Based on https://github.com/ducha-aiki/local_feature_tutorial repo with some modification.

__all__ = ['LocalFeatureExtractor', 'DescriptorMatcher', 'GeometricVerifier', 'SIFT', 'HardNetDesc', 'SNNMatcher',
           'SNNMMatcher', 'CV2_RANSACVerifier', 'TwoViewMatcher', 'degensac_Verifier', 'UAVPatchesANDPlus']

import cv2
import numpy as np
import abc
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from tqdm.notebook import tqdm
import numpy as np
import torch
import torchvision as tv
import os
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import sys, gc
from copy import deepcopy
import math
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import PIL

class LocalFeatureExtractor():
    '''Abstract method for local detector and or descriptor'''
    def __init__(self, **kwargs):
        ''''''
        return
    @abc.abstractmethod
    def detect(self, image: np.array, mask:np.array) -> List[cv2.KeyPoint]:
        return
    @abc.abstractmethod
    def compute(self, image: np.array, keypoints = None) -> Tuple[List[cv2.KeyPoint], np.array]:
        return

#export
class DescriptorMatcher():
    '''Abstract method fordescriptor matcher'''
    def __init__(self, **kwargs):
        return
    @abc.abstractmethod
    def match(self, queryDescriptors:np.array, trainDescriptors:np.array) -> List[cv2.DMatch]:
        out = []
        return out
#export
class GeometricVerifier():
    '''Abstract method for RANSAC'''
    def __init__(self, **kwargs):
        '''In'''
        return
    @abc.abstractmethod
    def verify(self, srcPts:np.array, dstPts:np.array):
        out = []
        return F, mask

# Cell
class SIFT(LocalFeatureExtractor):
    def __init__(self, **kwargs):
        self.desc = cv2.SIFT_create(40000)
        '''In'''
        return
    def compute(self, image: np.array, keypoints = None) -> Tuple[List[cv2.KeyPoint], np.array]:
        kps, descs  = self.desc.compute(image, keypoints)
        return kps, descs

# Cell
class HardNetDesc(LocalFeatureExtractor):
    def __init__(self, ModelT, **kwargs):
        self.Model = ModelT
        return
    def compute(self, image: np.array, keypoints = None) -> Tuple[List[cv2.KeyPoint], np.array]:
        import kornia.feature as KF, torch, torch.nn.functional as F
        from extract_patches.core import extract_patches
        # model = KF.HardNet(True).eval()
        # model = KF.HardNet(False).eval()
        if self.Model is not None:
          model = self.Model
        else:
          model = KF.HardNet(False).eval()
          model.load_state_dict(torch.load('/content/checkpoint_liberty_no_aug.pth', map_location=torch.device('cpu'))['state_dict'])
        patches = extract_patches(keypoints, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 32, 12.0)

        import torch
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        model = model.to(dev)
        torch_patches = torch.from_numpy(np.stack(patches, axis=0)).float().to(dev)#.cuda()
        torch_patches = torch_patches.unsqueeze(1)
        for idx in range(torch_patches.shape[0]):
          torch_patches[idx] = torch_patches[idx] / torch_patches[idx].max()
        out_desc = np.zeros((len(torch_patches), 128))
        bs = 2048
        for i in range(0, len(patches), bs):
            data_a = torch_patches[i: i + bs, :, :, :]
            with torch.no_grad():
                out_a = model(data_a)
            out_desc[i: i + bs,:] = out_a.data.cpu().numpy().reshape(-1, 128)
        return keypoints, out_desc

class UAVPatchesANDPlus(LocalFeatureExtractor):
    def __init__(self, ModelT, **kwargs):
        self.Model = ModelT
        return
    def compute(self, image: np.array, keypoints = None) -> Tuple[List[cv2.KeyPoint], np.array]:
        import kornia.feature as KF, torch, torch.nn.functional as F
        from extract_patches.core import extract_patches
        # model = KF.HardNet(True).eval()
        model = self.Model
        patches = extract_patches(keypoints, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 32, 12.0)

        import torch
        # dev = torch.device('cpu')
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        model = model.to(dev)
        torch_patches = torch.from_numpy(np.stack(patches, axis=0)).float().to(dev)#.cuda()
        torch_patches = torch_patches.unsqueeze(1)
        for idx in range(torch_patches.shape[0]):
          torch_patches[idx] = torch_patches[idx] / torch_patches[idx].max()
        out_desc = np.zeros((len(torch_patches), 128))
        bs = 2048
        for i in range(0, len(patches), bs):
            data_a = torch_patches[i: i + bs, :, :, :]
            with torch.no_grad():
                out_a = model(data_a)
            out_desc[i: i + bs,:] = out_a.data.cpu().numpy().reshape(-1, 128)
        return keypoints, out_desc
# Cell
class SNNMatcher():
    def __init__(self, th = 0.8):
        '''In'''
        self.th = th
        return
    def match(self, queryDescriptors:np.array, trainDescriptors:np.array) -> List[cv2.DMatch]:
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(queryDescriptors, trainDescriptors, 2)
        good_matches = []
        for m in matches:
            if m[0].distance / (1e-8 + m[1].distance) <= self.th :
                good_matches.append(m[0])
        return good_matches

# Cell
class SNNMMatcher():
    def __init__(self, th = 0.9):
        '''In'''
        self.th = th
        return
    def match(self, queryDescriptors:np.array, trainDescriptors:np.array) -> List[cv2.DMatch]:
        import kornia.feature as KF
        import torch
        # dev = torch.device('cpu')
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')

        dists, idxs = KF.match_smnn(torch.from_numpy(queryDescriptors).float().to(dev),
                                   torch.from_numpy(trainDescriptors).float().to(dev),
                                   self.th)
        good_matches = []
        for idx_q_t in idxs.detach().cpu().numpy():
            good_matches.append(cv2.DMatch(idx_q_t[0].item(), idx_q_t[1].item(), 0))
        return good_matches, dists

# Cell
class CV2_RANSACVerifier(GeometricVerifier):
    def __init__(self, th = 0.5):
        self.th = th
        return
    def verify(self, srcPts:np.array, dstPts:np.array):
        F, mask = cv2.findFundamentalMat(srcPts, dstPts, cv2.RANSAC, self.th)
        return F, mask

# Cell
class TwoViewMatcher():
    def __init__(self, detector:LocalFeatureExtractor = cv2.SIFT_create(8000),
                       descriptor:LocalFeatureExtractor = cv2.SIFT_create(8000),
                       matcher: DescriptorMatcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED),
                       geom_verif: GeometricVerifier = CV2_RANSACVerifier(0.5)):
        self.det = detector
        self.desc = descriptor
        self.matcher = matcher
        self.geom_verif = geom_verif
        return
    def verify(self, img1_fname, img2_fname):
        if type(img1_fname) is str:
            img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
        else:
            img1 = img1_fname
        if type(img2_fname) is str:
            img2 = cv2.cvtColor(cv2.imread(img2_fname), cv2.COLOR_BGR2RGB)
        else:
            img2 = img2_fname


        kps1 = self.det.detect(img1, None)
        kps1, descs1 = self.desc.compute(img1,  kps1)

        kps2 = self.det.detect(img2, None)
        kps2, descs2 = self.desc.compute(img2, kps2)

        tentative_matches, dists = self.matcher.match(descs1, descs2)

        src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1,2)

        F, mask = self.geom_verif.verify(src_pts, dst_pts)

        good_kpts1 = [ kps1[m.queryIdx] for i,m in enumerate(tentative_matches) if mask[i]]
        good_kpts2 = [ kps2[m.trainIdx] for i,m in enumerate(tentative_matches) if mask[i]]
        result = {'match_kpts1': good_kpts1,
                  'match_kpts2': good_kpts2,
                  'F': F,
                  'num_inl': len(good_kpts1),
                  'dists': dists[mask].detach().cpu().squeeze().numpy()}
        return result

import pydegensac
class degensac_Verifier(GeometricVerifier):
    def __init__(self, th = 0.5):
        self.th = th
        return
    def verify(self, srcPts:np.array, dstPts:np.array):
        F, mask = pydegensac.findFundamentalMatrix(srcPts, dstPts, self.th, 0.999, max_iters=250000)
        return F, mask
