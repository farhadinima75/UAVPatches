# Based on https://github.com/ducha-aiki/local_feature_tutorial repo with some modification.

__all__ = ['LocalFeatureExtractor', 'DescriptorMatcher', 'GeometricVerifier', 'SNNMatcher',
           'SNNMMatcher', 'CV2_RANSACVerifier', 'TwoViewMatcher', 'degensac_Verifier',
           'SIFT_SIFT', 'SIFT_HARDNET','SIFT_SOSNET', 'SIFT_L2NET','SIFT_ROOT_SIFT', 'SIFT_GEODESC', 'SIFT_TFEAT', 'SIFT_BROWN6',
           'SIFT_UAVPatches', 'SIFT_UAVPatchesPlus','CONTEXTDESC_CONTEXTDESC', 'D2NET_D2NET','R2D2_R2D2', 'KEYNET_KEYNET', 'ORB2_ORB2',
           'AKAZE_AKAZE', 'SUPERPOINT_SUPERPOINT', 'LFNET_LFNET', 'SIFT_LOGPOLAR', 'MSER_HARDNET', 'DISK_DISK', 'DELF_DELF', 'SIFT_VGG',
           'SIFT_DAISY', 'SIFT_BOOST_DESC', 'SIFT_LATCH', 'SIFT_FREAK', 'ORB2_UAVPatchesPlus', 'ORB2_UAVPatches', 'ORB2_BROWN6', 'ORB2_HARDNET',
           'ORB2_SOSNET', 'SIFT_MKDDescriptor', 'ORB2_MKDDescriptor', 'SIFT_HyNet', 'ORB2_HyNet']

import cv2, shutil
import numpy as np
import abc
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from tqdm.notebook import tqdm
import numpy as np, time
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
import kornia.feature as KF, torch, torch.nn.functional as F
from extract_patches.core import extract_patches
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt

from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes
from feature_manager import feature_manager_factory
from feature_manager_configs import FeatureManagerConfigs
from utils_features import descriptor_sigma_mad, compute_hom_reprojection_error
from utils_img import rotate_img, transform_img

def match_snn(desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8, dm: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(desc1.shape) != 2:
        raise AssertionError
    if len(desc2.shape) != 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError
    valsList, idxs_in_2List = [], []
    for Batch in range(0, len(desc1), 10000):
      MiniDesc = desc1[Batch:Batch+10000]
      dm = torch.cdist(MiniDesc, desc2)
      vals, idxs_in_2 = torch.topk(dm, 2, dim=1, largest=False)
      valsList.append(vals)
      idxs_in_2List.append(idxs_in_2)
      dm, vals, idxs_in_2 = [], [], []
    vals, idxs_in_2 = torch.cat(valsList), torch.cat(idxs_in_2List)
    ratio = vals[:, 0] / vals[:, 1]
    mask = ratio <= th
    match_dists = ratio[mask]
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=desc1.device)[mask]
    idxs_in_2 = idxs_in_2[:, 0][mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)

def match_smnn(
    desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8, dm: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if len(desc1.shape) != 2:
        raise AssertionError
    if len(desc2.shape) != 2:
        raise AssertionError
    if desc1.shape[0] < 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError
    
    dists1, idx1 = match_snn(desc1, desc2, th)
    gc.collect()
    dists2, idx2 = match_snn(desc2, desc1, th)

    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        idxs_dm = torch.cdist(idx1.float(), idx2.float(), p=1.0)
        mutual_idxs1 = idxs_dm.min(dim=1)[0] < 1e-8
        mutual_idxs2 = idxs_dm.min(dim=0)[0] < 1e-8
        good_idxs1 = idx1[mutual_idxs1.view(-1)]
        good_idxs2 = idx2[mutual_idxs2.view(-1)]
        dists1_good = dists1[mutual_idxs1.view(-1)]
        dists2_good = dists2[mutual_idxs2.view(-1)]
        _, idx_upl1 = torch.sort(good_idxs1[:, 0])
        _, idx_upl2 = torch.sort(good_idxs2[:, 0])
        good_idxs1 = good_idxs1[idx_upl1]
        match_dists = torch.max(dists1_good[idx_upl1], dists2_good[idx_upl2])
        matches_idxs = good_idxs1
    else:
        matches_idxs, match_dists = torch.empty(0, 2, device=desc1.device), torch.empty(0, 1, device=desc1.device)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)
           
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
def SIFT_SIFT(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.SIFT)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)

def SIFT_AKAZE(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.AKAZE)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)

def SIFT_ROOT_SIFT(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.ROOT_SIFT)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)

def SIFT_FREAK(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.FREAK)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)

def SIFT_HARDNET(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.HARDNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature 

def SIFT_LOGPOLAR(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.LOGPOLAR)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    # Feature._feature_descriptor.mag_factor = 12
    return Feature 

def SIFT_L2NET(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.L2NET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)

def SIFT_SOSNET(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.SOSNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature 

def SIFT_TFEAT(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.TFEAT)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature) 

def SIFT_GEODESC(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.GEODESC)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def SIFT_UAVPatches(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.UAVPatches)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature 

def SIFT_MKDDescriptor(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.MKDDescriptor)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature 

def SIFT_HyNet(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.HyNet)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature 

def SIFT_UAVPatchesPlus(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.UAVPatchesPlus)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature     

def SIFT_BROWN6(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.BROWN6)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature)
    Feature._feature_descriptor.mag_factor = 12
    return Feature       

def CONTEXTDESC_CONTEXTDESC(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 4,                                  
                   scale_factor = 1.2, 
                   detector_type   = FeatureDetectorTypes.CONTEXTDESC, 
                   descriptor_type = FeatureDescriptorTypes.CONTEXTDESC)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def D2NET_D2NET(num_features):
    Feature = dict(num_features    = num_features,   
                   detector_type   = FeatureDetectorTypes.D2NET, 
                   descriptor_type = FeatureDescriptorTypes.D2NET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_detector.max_edge = 3000
    Feature._feature_detector.max_sum_edges = 6000
    return Feature

def R2D2_R2D2(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.R2D2, 
                   descriptor_type = FeatureDescriptorTypes.R2D2)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_detector.repeatability_thr = 0.7 #default 
    Feature._feature_detector.reliability_thr = 0.7 #default 
    return Feature

def KEYNET_KEYNET(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 1,                                  
                   scale_factor = 1.2, 
                   detector_type   = FeatureDetectorTypes.KEYNET, 
                   descriptor_type = FeatureDescriptorTypes.KEYNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def AKAZE_AKAZE(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8,
                   detector_type   = FeatureDetectorTypes.AKAZE, 
                   descriptor_type = FeatureDescriptorTypes.AKAZE)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def SUPERPOINT_SUPERPOINT(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 1, 
                   scale_factor = 1.2,
                   detector_type   = FeatureDetectorTypes.SUPERPOINT, 
                   descriptor_type = FeatureDescriptorTypes.SUPERPOINT)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def LFNET_LFNET(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.LFNET, 
                   descriptor_type = FeatureDescriptorTypes.LFNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def ORB2_ORB2(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.ORB2)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def DELF_DELF(num_features):
    Feature = dict(num_features    = num_features, 
                   detector_type   = FeatureDetectorTypes.DELF, 
                   descriptor_type = FeatureDescriptorTypes.DELF)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def DISK_DISK(num_features):
    Feature = dict(num_features    = num_features, 
                   num_levels = 1,                                  
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.DISK, 
                   descriptor_type = FeatureDescriptorTypes.DISK)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    return Feature

def MSER_HARDNET(num_features):
    Feature = dict(num_features    = num_features, 
                   detector_type   = FeatureDetectorTypes.MSER, 
                   descriptor_type = FeatureDescriptorTypes.HARDNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def SIFT_VGG(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.VGG)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def SIFT_DAISY(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.DAISY)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def SIFT_BOOST_DESC(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.BOOST_DESC)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def SIFT_LATCH(num_features):
    Feature = dict(num_features    = num_features,
                   detector_type   = FeatureDetectorTypes.SIFT, 
                   descriptor_type = FeatureDescriptorTypes.LATCH)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    return feature_manager_factory(**Feature)  

def ORB2_MKDDescriptor(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.MKDDescriptor)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

def ORB2_HyNet(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.HyNet)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

def ORB2_UAVPatchesPlus(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.UAVPatchesPlus)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

def ORB2_UAVPatches(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.UAVPatches)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

def ORB2_HARDNET(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.HARDNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

def ORB2_BROWN6(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.BROWN6)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

def ORB2_SOSNET(num_features):
    Feature = dict(num_features    = num_features,
                   num_levels = 8, 
                   scale_factor = 1.2,   
                   detector_type   = FeatureDetectorTypes.ORB2, 
                   descriptor_type = FeatureDescriptorTypes.SOSNET)  
    Feature = FeatureManagerConfigs.extract_from(Feature)
    Feature = feature_manager_factory(**Feature) 
    Feature._feature_descriptor.mag_factor = 1
    return Feature

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
        # dev = torch.device('cpu')
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')

        dists, idxs = match_smnn(torch.from_numpy(queryDescriptors).float().to(dev),
                                   torch.from_numpy(trainDescriptors).float().to(dev),
                                   self.th)
        good_matches = []
        for idx_q_t in idxs.detach().cpu().numpy():
            good_matches.append(cv2.DMatch(idx_q_t[0].item(), idx_q_t[1].item(), 0))
        return good_matches, dists

class CV2_RANSACVerifier(GeometricVerifier):
    def __init__(self, th = 0.5):
        self.th = th
        return
    def verify(self, srcPts:np.array, dstPts:np.array):
        F, mask = cv2.findFundamentalMat(srcPts, dstPts, cv2.RANSAC, self.th)
        return F, mask

def WriteKeypoints(ImgPath, KeyPoints):
  Descs = np.arange(0,128)
  with open(ImgPath + '.txt', 'w') as F:
    F.write('{:d} 128\n'.format(len(KeyPoints)))
    for i, Key in enumerate(KeyPoints):
      F.write('{:f} {:f} {:f} {:f}'.format(Key.pt[0], Key.pt[1], Key.size, Key.angle))
      for D in Descs:
        F.write(' {:d}'.format(D))
      F.write('\n') 

class TwoViewMatcher():
    def __init__(self, detector_descriptor:LocalFeatureExtractor = cv2.SIFT_create(8000),
                       matcher: DescriptorMatcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED),
                       geom_verif: GeometricVerifier = CV2_RANSACVerifier(0.5)):
        self.detector_descriptor = detector_descriptor
        self.matcher = matcher
        self.geom_verif = geom_verif
        return
    def verify(self, img1_fname, img2_fname, kps1=None, kps2=None, Rotate=0, MaxSumImgSize=3000):
        if type(img1_fname) is str and kps1 is None:
            img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
        else:
            img1 = img1_fname
        if type(img2_fname) is str and kps2 is None:
            img2 = cv2.cvtColor(cv2.imread(img2_fname), cv2.COLOR_BGR2RGB)
        else:
            img2 = img2_fname

        if Rotate > 0: 
          img2, img2_box, M = rotate_img(img2, angle=Rotate, scale=1.0)  # rotation and scale    
          # img2, img2_box, H2 = transform_img(img1, rotx=20, roty=20, rotz=20, tx=1, ty=1, scale=1.2, adjust_frame=True)

        while np.sum(img1.shape[:-1]) > MaxSumImgSize: img1 = cv2.resize(img1, (int(img1.shape[1]*0.8), int(img1.shape[0]*0.8)))
        while np.sum(img2.shape[:-1]) > MaxSumImgSize: img2 = cv2.resize(img2, (int(img2.shape[1]*0.8), int(img2.shape[0]*0.8)))

        if kps1 == None:
          kps1 = self.detector_descriptor.detect(img1)
        kps1, descs1 = self.detector_descriptor.compute(img1,  kps1)

        T1 = time.time()
        if kps2 == None:
          kps2 = self.detector_descriptor.detect(img2)
        kps2, descs2 = self.detector_descriptor.compute(img2, kps2)
        T2 = time.time()
        
        tentative_matches, dists = self.matcher.match(descs1, descs2)

        minDim = np.min([descs1.shape[0], descs2.shape[0]])
        SigmaMad, _ = descriptor_sigma_mad(descs1[:minDim], descs2[:minDim])

        src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentative_matches]).reshape(-1,2)
        dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentative_matches]).reshape(-1,2)

        H, mask = self.geom_verif.verify(src_pts, dst_pts, H=True)

        good_kpts1 = [ kps1[m.queryIdx] for i,m in enumerate(tentative_matches) if mask[i]]
        good_kpts2 = [ kps2[m.trainIdx] for i,m in enumerate(tentative_matches) if mask[i]]

        good_kpts1pt, good_kpts2pt = np.float32([K.pt for K in good_kpts1]), np.float32([K.pt for K in good_kpts2])
        H, maskH = self.geom_verif.verify(good_kpts1pt, good_kpts2pt, H=True)
        ReprojectionError = compute_hom_reprojection_error(H, np.float32([K.pt for K in good_kpts2]), 
                                                               np.float32([K.pt for K in good_kpts1]))

        dists = dists.detach().cpu().squeeze().numpy()  
        Precision, Recall, Threshold = precision_recall_curve(np.array(mask)*1, 1 - dists)
        AveragePrecision = average_precision_score(np.array(mask)*1, 1 - dists)

        WriteKeypoints(img1_fname, kps1)
        WriteKeypoints(img2_fname, kps2)
        with open(f"{img1_fname}.Colmap_Matches_1to2.txt", 'w') as F:
          F.write('{} {}\n'.format(img1_fname.split('/')[-1], img2_fname.split('/')[-1]))
          for J, M in enumerate(tentative_matches):
            if mask[J]:
              F.write('{} {}\n'.format(M.queryIdx, M.trainIdx))
        os.system('colmap database_creator --database_path "{}".db'.format(img1_fname))
        KeyPath = '/'.join(img1_fname.split('/')[:-1])
        os.system('colmap feature_importer --database_path "{}".db --import_path "{}" --image_path "{}"'.format(img1_fname, KeyPath, KeyPath))
        MatchesPath = f"{img1_fname}.Colmap_Matches_1to2.txt"
        os.system('colmap matches_importer --database_path "{}".db --match_list_path "{}" --match_type inliers'.format(img1_fname, MatchesPath))
        os.makedirs(os.path.join(KeyPath, 'SFM'), exist_ok=True)
        os.system('colmap mapper --database_path "{}".db --image_path "{}" \
                                 --output_path "{}" --Mapper.tri_ignore_two_view_tracks 0 \
                                 --Mapper.filter_max_reproj_error 4 \
                                 --Mapper.filter_min_tri_angle 1.5 \
                                 --Mapper.init_min_num_inliers 15 \
                                 --Mapper.init_max_error 30 \
                                 --Mapper.init_max_forward_motion 0.99999999999999996 \
                                 --Mapper.init_max_reg_trials 6 \
                                 --Mapper.init_min_tri_angle 1'.format(img1_fname, KeyPath, os.path.join(KeyPath, 'SFM')))
        O = os.popen('colmap model_analyzer --path "{}"'.format(os.path.join(KeyPath, 'SFM/0'))).read().split('\n')
        os.system('colmap point_filtering --input_path "{}" \
                                          --output_path "{}" \
                                          --max_reproj_error 0.25'.format(os.path.join(KeyPath, 'SFM/0'), os.path.join(KeyPath, 'SFM/0')))
        O2 = os.popen('colmap model_analyzer --path "{}"'.format(os.path.join(KeyPath, 'SFM/0'))).read().split('\n')
        if os.path.isdir(os.path.join(KeyPath, 'SFM')): shutil.rmtree(os.path.join(KeyPath, 'SFM'))

        print(f'\033[92m3 x sigma-MAD of descriptor distances: {3*SigmaMad:.4f}\033[0m')
        print(f'\033[92mHemographic reprojection error: {ReprojectionError:.4f}\033[0m')
        print(f'\033[92mAverage Precision: {AveragePrecision*100:.4f}\033[0m')
        print(f'\033[92mFinal Matches: {len(good_kpts1)}\033[0m')
        print(f'\033[92mInliers Ratio: {float(len(good_kpts1))/float(len(src_pts)):.4f}\033[0m')
        print(f'\033[92mDetector and Descriptor Time: {T2 - T1:.2f}\033[0m')
        SFMReprojectionError, SFMReprojectionErrorBelow0_25, Points, PointsBelow0_25, C = 0, 0, 0, 0, 0
        for o in O:
          if 'reprojection' in o or 'Points' in o: 
            if 'reprojection' in o: SFMReprojectionError = float(o.split(' ')[-1][:-2])
            if 'Points' in o: Points = int(o.split(' ')[-1])
            print(f'\033[1m\033[96mColmap SFM {o}\033[0m')
            C+=1
        for o in O2:
          if 'reprojection' in o or 'Points' in o: 
            if 'reprojection' in o: SFMReprojectionErrorBelow0_25 = float(o.split(' ')[-1][:-2])
            if 'Points' in o: PointsBelow0_25 = int(o.split(' ')[-1])
            print(f'\033[1m\033[96mColmap SFM (Below 0.25 pix Error) {o}\033[0m')
            C+=1
        if C == 0: print('\033[1m\033[96mColmap SFM Failed. Normally, increasing MaxSumImgSize or TotalKeyPoints will solve the problem.\033[0m')
        
        result = {'img1': img1, 'img2': img2,
                  'init_kpts1': kps1,
                  'init_kpts2': kps2,
                  'match_kpts1': good_kpts1,
                  'match_kpts2': good_kpts2,
                  'H': H,
                  'ReprojectionError': ReprojectionError,
                  'AveragePrecision': AveragePrecision,
                  'Precision': Precision,
                  'Recall': Recall,
                  'num_inl': len(good_kpts1),
                  'dists': dists[mask],
                  'DetDescTime': T2 - T1,
                  'TentativeMatches': tentative_matches,
                  'InliersRatio': float(len(good_kpts1))/float(len(src_pts)),
                  'SFMReprojectionError': SFMReprojectionError,
                  'SFMReprojectionErrorBelow0_25': SFMReprojectionErrorBelow0_25,
                  'Points': Points,
                  'PointsBelow0_25': PointsBelow0_25}
        return result

# Cell
import pydegensac
class degensac_Verifier(GeometricVerifier):
    def __init__(self, th = 0.5):
        self.th = th
        return
    def verify(self, srcPts:np.array, dstPts:np.array, H=False):
        if H:
          H, mask = pydegensac.findHomography(dstPts, srcPts, self.th, 0.9999999, max_iters=250000)
          return H, mask
        F, mask = pydegensac.findFundamentalMatrix(srcPts, dstPts, self.th, 0.9999999, max_iters=250000)
        return F, mask
