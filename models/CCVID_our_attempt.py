import shutil

import cv2
import pandas as pd

from FaceDetection.faceDetector import FaceDetector, is_img
from FaceDetection.arcface import ArcFace
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm
from distutils.dir_util import copy_tree
from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis

CCVID_DATA_PATH = '/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID/'
OUR_CCVID_PATH = '/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID_our_attempt/insightface_detector'
OUR_CCVID_FAST_DATASET = '/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID_our_attempt/fast_reid_with_unsup'
CCVID_THRESHOLDS = '/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID_our_attempt/CCVID_thresholds_insightface_detector.csv'


'''
Gallery sequences:
'''
# GALLERY_SEQUENCES = ['session3/001_01', 'session3/001_02', 'session3/001_03', 'session3/001_04', 'session3/001_05',
#                      'session3/001_06', 'session3/001_07', 'session3/001_08', 'session3/001_09', 'session3/001_10',
#                      'session3/001_11', 'session3/001_12', 'session3/002_01', 'session3/002_02', 'session3/002_03',
#                      'session3/002_04', 'session3/002_05', 'session3/002_06', 'session3/002_07', 'session3/002_08',
#                      'session3/002_09', 'session3/002_10', 'session3/002_11', 'session3/002_12', 'session3/004_01',
#                      'session3/004_02', 'session3/004_03', 'session3/004_04', 'session3/004_05', 'session3/004_06',
#                      'session3/004_07', 'session3/004_08', 'session3/004_09', 'session3/004_10', 'session3/004_11',
#                      'session3/004_12', 'session3/007_01', 'session3/007_02', 'session3/007_03', 'session3/007_04',
#                      'session3/007_05', 'session3/007_06', 'session3/007_07', 'session3/007_08', 'session3/007_09',
#                      'session3/007_10', 'session3/007_11', 'session3/007_12', 'session3/008_01', 'session3/008_02',
#                      'session3/008_03', 'session3/008_04', 'session3/008_05', 'session3/008_06', 'session3/008_07',
#                      'session3/008_08', 'session3/008_09', 'session3/008_10', 'session3/008_11', 'session3/008_12',
#                      'session3/012_01', 'session3/012_02', 'session3/012_03', 'session3/012_04', 'session3/012_05',
#                      'session3/012_06', 'session3/012_07', 'session3/012_08', 'session3/012_09', 'session3/012_10',
#                      'session3/012_11', 'session3/012_12', 'session3/013_01', 'session3/013_02', 'session3/013_03',
#                      'session3/013_04', 'session3/013_05', 'session3/013_06', 'session3/013_07', 'session3/013_08',
#                      'session3/013_09', 'session3/013_10', 'session3/013_11', 'session3/013_12', 'session3/017_01',
#                      'session3/017_02', 'session3/017_03', 'session3/017_04', 'session3/017_05', 'session3/017_06',
#                      'session3/017_07', 'session3/017_08', 'session3/017_09', 'session3/017_10', 'session3/017_11',
#                      'session3/017_12', 'session1/003_04', 'session1/003_05', 'session1/003_06', 'session1/003_07',
#                      'session1/003_08', 'session1/003_09', 'session1/003_10', 'session1/003_11', 'session1/003_12',
#                      'session1/005_04', 'session1/005_05', 'session1/005_06', 'session1/005_07', 'session1/005_08',
#                      'session1/005_09', 'session1/005_10', 'session1/005_11', 'session1/005_12']
GALLERY_SEQUENCES = ['session3/001_01', 'session3/001_02', 'session3/001_03', 'session3/001_04', 'session3/001_05',
                     'session3/001_06', 'session3/001_07', 'session3/001_08', 'session3/001_09', 'session3/001_10',
                     'session3/001_11', 'session3/001_12', 'session3/002_01', 'session3/002_02', 'session3/002_03',
                     'session3/002_04', 'session3/002_05', 'session3/002_06', 'session3/002_07', 'session3/002_08',
                     'session3/002_09', 'session3/002_10', 'session3/002_11', 'session3/002_12', 'session3/004_01',
                     'session3/004_02', 'session3/004_03', 'session3/004_04', 'session3/004_05', 'session3/004_06',
                     'session3/004_07', 'session3/004_08', 'session3/004_09', 'session3/004_10', 'session3/004_11',
                     'session3/004_12', 'session3/007_01', 'session3/007_02', 'session3/007_03', 'session3/007_04',
                     'session3/007_05', 'session3/007_06', 'session3/007_07', 'session3/007_08', 'session3/007_09',
                     'session3/007_10', 'session3/007_11', 'session3/007_12', 'session3/008_01', 'session3/008_02',
                     'session3/008_03', 'session3/008_04', 'session3/008_05', 'session3/008_06', 'session3/008_07',
                     'session3/008_08', 'session3/008_09', 'session3/008_10', 'session3/008_11', 'session3/008_12',
                     'session3/012_01', 'session3/012_02', 'session3/012_03', 'session3/012_04', 'session3/012_05',
                     'session3/012_06', 'session3/012_07', 'session3/012_08', 'session3/012_09', 'session3/012_10',
                     'session3/012_11', 'session3/012_12', 'session3/013_01', 'session3/013_02', 'session3/013_03',
                     'session3/013_04', 'session3/013_05', 'session3/013_06', 'session3/013_07', 'session3/013_08',
                     'session3/013_09', 'session3/013_10', 'session3/013_11', 'session3/013_12', 'session3/017_01',
                     'session3/017_02', 'session3/017_03', 'session3/017_04', 'session3/017_05', 'session3/017_06',
                     'session3/017_07', 'session3/017_08', 'session3/017_09', 'session3/017_10', 'session3/017_11',
                     'session3/017_12', 'session1/003_04', 'session1/003_05', 'session1/003_06', 'session1/003_07',
                     'session1/003_08', 'session1/003_09', 'session1/003_10', 'session1/003_11', 'session1/003_12',
                     'session1/005_04', 'session1/005_05', 'session1/005_06', 'session1/005_07', 'session1/005_08',
                     'session1/005_09', 'session1/005_10', 'session1/005_11', 'session1/005_12', 'session1/006_04',
                     'session1/006_05', 'session1/006_06', 'session1/006_07', 'session1/006_08', 'session1/006_09',
                     'session1/006_10', 'session1/006_11', 'session1/006_12', 'session1/009_04', 'session1/009_05',
                     'session1/009_06', 'session1/009_07', 'session1/009_08', 'session1/009_09', 'session1/009_10',
                     'session1/009_11', 'session1/009_12', 'session1/010_04', 'session1/010_05', 'session1/010_06',
                     'session1/010_07', 'session1/010_08', 'session1/010_09', 'session1/010_10', 'session1/010_11',
                     'session1/010_12', 'session1/011_04', 'session1/011_05', 'session1/011_06', 'session1/011_07',
                     'session1/011_08', 'session1/011_09', 'session1/011_10', 'session1/011_11', 'session1/011_12',
                     'session1/014_04', 'session1/014_05', 'session1/014_06', 'session1/014_07', 'session1/014_08',
                     'session1/014_09', 'session1/014_10', 'session1/014_11', 'session1/014_12', 'session1/015_04',
                     'session1/015_05', 'session1/015_06', 'session1/015_07', 'session1/015_08', 'session1/015_09',
                     'session1/015_10', 'session1/015_11', 'session1/015_12', 'session1/016_04', 'session1/016_05',
                     'session1/016_06', 'session1/016_07', 'session1/016_08', 'session1/016_09', 'session1/016_10',
                     'session1/016_11', 'session1/016_12', 'session1/018_04', 'session1/018_05', 'session1/018_06',
                     'session1/018_07', 'session1/018_08', 'session1/018_09', 'session1/018_10', 'session1/018_11',
                     'session1/018_12', 'session1/019_04', 'session1/019_05', 'session1/019_06', 'session1/019_07',
                     'session1/019_08', 'session1/019_09', 'session1/019_10', 'session1/019_11', 'session1/019_12',
                     'session1/020_04', 'session1/020_05', 'session1/020_06', 'session1/020_07', 'session1/020_08',
                     'session1/020_09', 'session1/020_10', 'session1/020_11', 'session1/020_12', 'session1/021_04',
                     'session1/021_05', 'session1/021_06', 'session1/021_07', 'session1/021_08', 'session1/021_09',
                     'session1/021_10', 'session1/021_11', 'session1/021_12', 'session1/022_04', 'session1/022_05',
                     'session1/022_06', 'session1/022_07', 'session1/022_08', 'session1/022_09', 'session1/022_10',
                     'session1/022_11', 'session1/022_12', 'session1/023_04', 'session1/023_05', 'session1/023_06',
                     'session1/023_07', 'session1/023_08', 'session1/023_09', 'session1/023_10', 'session1/023_11',
                     'session1/023_12', 'session1/024_04', 'session1/024_05', 'session1/024_06', 'session1/024_07',
                     'session1/024_08', 'session1/024_09', 'session1/024_10', 'session1/024_11', 'session1/024_12',
                     'session1/025_04', 'session1/025_05', 'session1/025_06', 'session1/025_07', 'session1/025_08',
                     'session1/025_09', 'session1/025_10', 'session1/025_11', 'session1/025_12', 'session1/026_04',
                     'session1/026_05', 'session1/026_06', 'session1/026_07', 'session1/026_08', 'session1/026_09',
                     'session1/026_10', 'session1/026_11', 'session1/026_12', 'session1/027_04', 'session1/027_05',
                     'session1/027_06', 'session1/027_07', 'session1/027_08', 'session1/027_09', 'session1/027_10',
                     'session1/027_11', 'session1/027_12', 'session1/028_04', 'session1/028_05', 'session1/028_06',
                     'session1/028_07', 'session1/028_08', 'session1/028_09', 'session1/028_10', 'session1/028_11',
                     'session1/028_12', 'session1/029_04', 'session1/029_05', 'session1/029_06', 'session1/029_07',
                     'session1/029_08', 'session1/029_09', 'session1/029_10', 'session1/029_11', 'session1/029_12',
                     'session1/030_04', 'session1/030_05', 'session1/030_06', 'session1/030_07', 'session1/030_08',
                     'session1/030_09', 'session1/030_10', 'session1/030_11', 'session1/030_12', 'session1/032_04',
                     'session1/032_05', 'session1/032_06', 'session1/032_07', 'session1/032_08', 'session1/032_09',
                     'session1/032_10', 'session1/032_11', 'session1/032_12', 'session1/033_04', 'session1/033_05',
                     'session1/033_06', 'session1/033_07', 'session1/033_08', 'session1/033_09', 'session1/033_10',
                     'session1/033_11', 'session1/033_12', 'session1/034_04', 'session1/034_05', 'session1/034_06',
                     'session1/034_07', 'session1/034_08', 'session1/034_09', 'session1/034_10', 'session1/034_11',
                     'session1/034_12', 'session1/035_04', 'session1/035_05', 'session1/035_06', 'session1/035_07',
                     'session1/035_08', 'session1/035_09', 'session1/035_10', 'session1/035_11', 'session1/035_12',
                     'session1/036_04', 'session1/036_05', 'session1/036_06', 'session1/036_07', 'session1/036_08',
                     'session1/036_09', 'session1/036_10', 'session1/036_11', 'session1/036_12', 'session1/037_04',
                     'session1/037_05', 'session1/037_06', 'session1/037_07', 'session1/037_08', 'session1/037_09',
                     'session1/037_10', 'session1/037_11', 'session1/037_12', 'session1/038_04', 'session1/038_05',
                     'session1/038_06', 'session1/038_07', 'session1/038_08', 'session1/038_09', 'session1/038_10',
                     'session1/038_11', 'session1/038_12', 'session1/039_04', 'session1/039_05', 'session1/039_06',
                     'session1/039_07', 'session1/039_08', 'session1/039_09', 'session1/039_10', 'session1/039_11',
                     'session1/039_12', 'session1/041_04', 'session1/041_05', 'session1/041_06', 'session1/041_07',
                     'session1/041_08', 'session1/041_09', 'session1/041_10', 'session1/041_11', 'session1/041_12',
                     'session1/042_04', 'session1/042_05', 'session1/042_06', 'session1/042_07', 'session1/042_08',
                     'session1/042_09', 'session1/042_10', 'session1/042_11', 'session1/042_12', 'session1/043_04',
                     'session1/043_05', 'session1/043_06', 'session1/043_07', 'session1/043_08', 'session1/043_09',
                     'session1/043_10', 'session1/043_11', 'session1/043_12', 'session1/044_04', 'session1/044_05',
                     'session1/044_06', 'session1/044_07', 'session1/044_08', 'session1/044_09', 'session1/044_10',
                     'session1/044_11', 'session1/044_12', 'session1/045_04', 'session1/045_05', 'session1/045_06',
                     'session1/045_07', 'session1/045_08', 'session1/045_09', 'session1/045_10', 'session1/045_11',
                     'session1/045_12', 'session1/046_04', 'session1/046_05', 'session1/046_06', 'session1/046_07',
                     'session1/046_08', 'session1/046_09', 'session1/046_10', 'session1/046_11', 'session1/046_12',
                     'session1/047_04', 'session1/047_05', 'session1/047_06', 'session1/047_07', 'session1/047_08',
                     'session1/047_09', 'session1/047_10', 'session1/047_11', 'session1/047_12', 'session1/049_04',
                     'session1/049_05', 'session1/049_06', 'session1/049_07', 'session1/049_08', 'session1/049_09',
                     'session1/049_10', 'session1/049_11', 'session1/049_12', 'session1/050_04', 'session1/050_05',
                     'session1/050_06', 'session1/050_07', 'session1/050_08', 'session1/050_09', 'session1/050_10',
                     'session1/050_11', 'session1/050_12', 'session1/051_04', 'session1/051_05', 'session1/051_06',
                     'session1/051_07', 'session1/051_08', 'session1/051_09', 'session1/051_10', 'session1/051_11',
                     'session1/051_12', 'session1/052_04', 'session1/052_05', 'session1/052_06', 'session1/052_07',
                     'session1/052_08', 'session1/052_09', 'session1/052_10', 'session1/052_11', 'session1/052_12',
                     'session1/053_04', 'session1/053_05', 'session1/053_06', 'session1/053_07', 'session1/053_08',
                     'session1/053_09', 'session1/053_10', 'session1/053_11', 'session1/053_12', 'session1/054_04',
                     'session1/054_05', 'session1/054_06', 'session1/054_07', 'session1/054_08', 'session1/054_09',
                     'session1/054_10', 'session1/054_11', 'session1/054_12', 'session1/055_04', 'session1/055_05',
                     'session1/055_06', 'session1/055_07', 'session1/055_08', 'session1/055_09', 'session1/055_10',
                     'session1/055_11', 'session1/055_12', 'session1/056_04', 'session1/056_05', 'session1/056_06',
                     'session1/056_07', 'session1/056_08', 'session1/056_09', 'session1/056_10', 'session1/056_11',
                     'session1/056_12', 'session1/057_04', 'session1/057_05', 'session1/057_06', 'session1/057_07',
                     'session1/057_08', 'session1/057_09', 'session1/057_10', 'session1/057_11', 'session1/057_12',
                     'session1/058_04', 'session1/058_05', 'session1/058_06', 'session1/058_07', 'session1/058_08',
                     'session1/058_09', 'session1/058_10', 'session1/058_11', 'session1/058_12', 'session1/059_04',
                     'session1/059_05', 'session1/059_06', 'session1/059_07', 'session1/059_08', 'session1/059_09',
                     'session1/059_10', 'session1/059_11', 'session1/059_12', 'session1/060_04', 'session1/060_05',
                     'session1/060_06', 'session1/060_07', 'session1/060_08', 'session1/060_09', 'session1/060_10',
                     'session1/060_11', 'session1/060_12', 'session1/061_04', 'session1/061_05', 'session1/061_06',
                     'session1/061_07', 'session1/061_08', 'session1/061_09', 'session1/061_10', 'session1/061_11',
                     'session1/061_12', 'session1/062_04', 'session1/062_05', 'session1/062_06', 'session1/062_07',
                     'session1/062_08', 'session1/062_09', 'session1/062_10', 'session1/062_11', 'session1/062_12',
                     'session1/063_04', 'session1/063_05', 'session1/063_06', 'session1/063_07', 'session1/063_08',
                     'session1/063_09', 'session1/063_10', 'session1/063_11', 'session1/063_12', 'session1/064_04',
                     'session1/064_05', 'session1/064_06', 'session1/064_07', 'session1/064_08', 'session1/064_09',
                     'session1/064_10', 'session1/064_11', 'session1/064_12', 'session1/065_04', 'session1/065_05',
                     'session1/065_06', 'session1/065_07', 'session1/065_08', 'session1/065_09', 'session1/065_10',
                     'session1/065_11', 'session1/065_12', 'session1/066_04', 'session1/066_05', 'session1/066_06',
                     'session1/066_07', 'session1/066_08', 'session1/066_09', 'session1/066_10', 'session1/066_11',
                     'session1/066_12', 'session1/067_04', 'session1/067_05', 'session1/067_06', 'session1/067_07',
                     'session1/067_08', 'session1/067_09', 'session1/067_10', 'session1/067_11', 'session1/067_12',
                     'session1/068_04', 'session1/068_05', 'session1/068_06', 'session1/068_07', 'session1/068_08',
                     'session1/068_09', 'session1/068_10', 'session1/068_11', 'session1/068_12', 'session1/069_04',
                     'session1/069_05', 'session1/069_06', 'session1/069_07', 'session1/069_08', 'session1/069_09',
                     'session1/069_10', 'session1/069_11', 'session1/069_12', 'session1/070_04', 'session1/070_05',
                     'session1/070_06', 'session1/070_07', 'session1/070_08', 'session1/070_09', 'session1/070_10',
                     'session1/070_11', 'session1/070_12', 'session1/071_04', 'session1/071_05', 'session1/071_06',
                     'session1/071_07', 'session1/071_08', 'session1/071_09', 'session1/071_10', 'session1/071_11',
                     'session1/071_12', 'session1/072_04', 'session1/072_05', 'session1/072_06', 'session1/072_07',
                     'session1/072_08', 'session1/072_09', 'session1/072_10', 'session1/072_11', 'session1/072_12',
                     'session1/073_04', 'session1/073_05', 'session1/073_06', 'session1/073_07', 'session1/073_08',
                     'session1/073_09', 'session1/073_10', 'session1/073_11', 'session1/073_12', 'session1/074_04',
                     'session1/074_05', 'session1/074_06', 'session1/074_07', 'session1/074_08', 'session1/074_09',
                     'session1/074_10', 'session1/074_11', 'session1/074_12', 'session1/075_04', 'session1/075_05',
                     'session1/075_06', 'session1/075_07', 'session1/075_08', 'session1/075_09', 'session1/075_10',
                     'session1/075_11', 'session1/075_12', 'session1/076_04', 'session1/076_05', 'session1/076_06',
                     'session1/076_07', 'session1/076_08', 'session1/076_09', 'session1/076_10', 'session1/076_11',
                     'session1/076_12', 'session1/078_04', 'session1/078_05', 'session1/078_06', 'session1/078_07',
                     'session1/078_08', 'session1/078_09', 'session1/078_10', 'session1/078_11', 'session1/078_12',
                     'session1/079_04', 'session1/079_05', 'session1/079_06', 'session1/079_07', 'session1/079_08',
                     'session1/079_09', 'session1/079_10', 'session1/079_11', 'session1/079_12', 'session1/080_04',
                     'session1/080_05', 'session1/080_06', 'session1/080_07', 'session1/080_08', 'session1/080_09',
                     'session1/080_10', 'session1/080_11', 'session1/080_12', 'session1/081_04', 'session1/081_05',
                     'session1/081_06', 'session1/081_07', 'session1/081_08', 'session1/081_09', 'session1/081_10',
                     'session1/081_11', 'session1/081_12', 'session1/082_04', 'session1/082_05', 'session1/082_06',
                     'session1/082_07', 'session1/082_08', 'session1/082_09', 'session1/082_10', 'session1/082_11',
                     'session1/082_12', 'session1/083_04', 'session1/083_05', 'session1/083_06', 'session1/083_07',
                     'session1/083_08', 'session1/083_09', 'session1/083_10', 'session1/083_11', 'session1/083_12',
                     'session1/084_04', 'session1/084_05', 'session1/084_06', 'session1/084_07', 'session1/084_08',
                     'session1/084_09', 'session1/084_10', 'session1/084_11', 'session1/084_12', 'session1/085_04',
                     'session1/085_05', 'session1/085_06', 'session1/085_07', 'session1/085_08', 'session1/085_09',
                     'session1/085_10', 'session1/085_11', 'session1/085_12', 'session1/086_04', 'session1/086_05',
                     'session1/086_06', 'session1/086_07', 'session1/086_08', 'session1/086_09', 'session1/086_10',
                     'session1/086_11', 'session1/086_12', 'session1/087_04', 'session1/087_05', 'session1/087_06',
                     'session1/087_07', 'session1/087_08', 'session1/087_09', 'session1/087_10', 'session1/087_11',
                     'session1/087_12', 'session1/088_04', 'session1/088_05', 'session1/088_06', 'session1/088_07',
                     'session1/088_08', 'session1/088_09', 'session1/088_10', 'session1/088_11', 'session1/088_12',
                     'session1/089_04', 'session1/089_05', 'session1/089_06', 'session1/089_07', 'session1/089_08',
                     'session1/089_09', 'session1/089_10', 'session1/089_11', 'session1/089_12', 'session1/090_04',
                     'session1/090_05', 'session1/090_06', 'session1/090_07', 'session1/090_08', 'session1/090_09',
                     'session1/090_10', 'session1/090_11', 'session1/090_12', 'session1/091_04', 'session1/091_05',
                     'session1/091_06', 'session1/091_07', 'session1/091_08', 'session1/091_09', 'session1/091_10',
                     'session1/091_11', 'session1/091_12', 'session1/092_04', 'session1/092_05', 'session1/092_06',
                     'session1/092_07', 'session1/092_08', 'session1/092_09', 'session1/092_10', 'session1/092_11',
                     'session1/092_12', 'session1/093_04', 'session1/093_05', 'session1/093_06', 'session1/093_07',
                     'session1/093_08', 'session1/093_09', 'session1/093_10', 'session1/093_11', 'session1/093_12',
                     'session1/094_04', 'session1/094_05', 'session1/094_06', 'session1/094_07', 'session1/094_08',
                     'session1/094_09', 'session1/094_10', 'session1/094_11', 'session1/094_12', 'session1/095_04',
                     'session1/095_05', 'session1/095_06', 'session1/095_07', 'session1/095_08', 'session1/095_09',
                     'session1/095_10', 'session1/095_11', 'session1/095_12', 'session1/096_04', 'session1/096_05',
                     'session1/096_06', 'session1/096_07', 'session1/096_08', 'session1/096_09', 'session1/096_10',
                     'session1/096_11', 'session1/096_12', 'session1/097_04', 'session1/097_05', 'session1/097_06',
                     'session1/097_07', 'session1/097_08', 'session1/097_09', 'session1/097_10', 'session1/097_11',
                     'session1/097_12', 'session1/098_04', 'session1/098_05', 'session1/098_06', 'session1/098_07',
                     'session1/098_08', 'session1/098_09', 'session1/098_10', 'session1/098_11', 'session1/098_12',
                     'session1/099_04', 'session1/099_05', 'session1/099_06', 'session1/099_07', 'session1/099_08',
                     'session1/099_09', 'session1/099_10', 'session1/099_11', 'session1/099_12', 'session1/100_04',
                     'session1/100_05', 'session1/100_06', 'session1/100_07', 'session1/100_08', 'session1/100_09',
                     'session1/100_10', 'session1/100_11', 'session1/100_12', 'session1/101_04', 'session1/101_05',
                     'session1/101_06', 'session1/101_07', 'session1/101_08', 'session1/101_09', 'session1/101_10',
                     'session1/101_11', 'session1/101_12', 'session1/102_04', 'session1/102_05', 'session1/102_06',
                     'session1/102_07', 'session1/102_08', 'session1/102_09', 'session1/102_10', 'session1/102_11',
                     'session1/102_12', 'session2/148_04', 'session2/148_05', 'session2/148_06', 'session2/148_07',
                     'session2/148_08', 'session2/148_09', 'session2/148_10', 'session2/148_11', 'session2/148_12',
                     'session2/149_07', 'session2/149_08', 'session2/149_09', 'session2/150_07', 'session2/150_08',
                     'session2/150_09', 'session2/151_07', 'session2/151_08', 'session2/151_09', 'session2/152_07',
                     'session2/152_08', 'session2/152_09', 'session2/153_07', 'session2/153_08', 'session2/153_09',
                     'session2/153_10', 'session2/153_11', 'session2/153_12', 'session2/154_07', 'session2/154_08',
                     'session2/154_09', 'session2/155_07', 'session2/155_08', 'session2/155_09', 'session2/156_07',
                     'session2/156_08', 'session2/156_09', 'session2/157_07', 'session2/157_08', 'session2/157_09',
                     'session2/158_07', 'session2/158_08', 'session2/158_09', 'session2/159_07', 'session2/159_08',
                     'session2/159_09', 'session2/160_07', 'session2/160_08', 'session2/160_09', 'session2/161_07',
                     'session2/161_08', 'session2/161_09', 'session2/162_07', 'session2/162_08', 'session2/162_09',
                     'session2/163_07', 'session2/163_08', 'session2/163_09', 'session2/164_07', 'session2/164_08',
                     'session2/164_09', 'session2/165_07', 'session2/165_08', 'session2/165_09', 'session2/166_07',
                     'session2/166_08', 'session2/166_09', 'session2/167_07', 'session2/167_08', 'session2/167_09',
                     'session2/168_07', 'session2/168_08', 'session2/168_09', 'session2/169_07', 'session2/169_08',
                     'session2/169_09', 'session2/170_07', 'session2/170_08', 'session2/170_09', 'session2/171_07',
                     'session2/171_08', 'session2/171_09', 'session2/172_07', 'session2/172_08', 'session2/172_09',
                     'session2/173_07', 'session2/173_08', 'session2/173_09', 'session2/174_07', 'session2/174_08',
                     'session2/174_09', 'session2/175_07', 'session2/175_08', 'session2/175_09', 'session2/176_07',
                     'session2/176_08', 'session2/176_09', 'session2/177_07', 'session2/177_08', 'session2/177_09',
                     'session2/178_07', 'session2/178_08', 'session2/178_09', 'session2/179_07', 'session2/179_08',
                     'session2/179_09', 'session2/180_07', 'session2/180_08', 'session2/180_09', 'session2/181_07',
                     'session2/181_08', 'session2/181_09', 'session2/182_07', 'session2/182_08', 'session2/182_09',
                     'session2/183_07', 'session2/183_08', 'session2/183_09', 'session2/184_07', 'session2/184_08',
                     'session2/184_09', 'session2/185_07', 'session2/185_08', 'session2/185_09', 'session2/186_07',
                     'session2/186_08', 'session2/186_09', 'session2/187_07', 'session2/187_08', 'session2/187_09',
                     'session2/188_07', 'session2/188_08', 'session2/188_09', 'session2/189_07', 'session2/189_08',
                     'session2/189_09', 'session2/190_07', 'session2/190_08', 'session2/190_09', 'session2/191_07',
                     'session2/191_08', 'session2/191_09', 'session2/192_07', 'session2/192_08', 'session2/192_09',
                     'session2/193_07', 'session2/193_08', 'session2/193_09', 'session2/194_07', 'session2/194_08',
                     'session2/194_09', 'session2/195_07', 'session2/195_08', 'session2/195_09', 'session2/196_07',
                     'session2/196_08', 'session2/196_09', 'session2/197_07', 'session2/197_08', 'session2/197_09',
                     'session2/198_07', 'session2/198_08', 'session2/198_09', 'session2/199_07', 'session2/199_08',
                     'session2/199_09', 'session2/200_07', 'session2/200_08', 'session2/200_09']


'''
Query sequences:
'''
# QUERY_SEQUENCES = ['session1/001_01', 'session1/001_02', 'session1/001_03', 'session1/001_04', 'session1/001_05',
#                    'session1/001_06', 'session1/001_07', 'session1/001_08', 'session1/001_09', 'session1/001_10',
#                    'session1/001_11', 'session1/001_12', 'session1/002_01', 'session1/002_02', 'session1/002_03',
#                    'session1/002_04', 'session1/002_05', 'session1/002_06', 'session1/002_07', 'session1/002_08',
#                    'session1/002_09', 'session1/002_10', 'session1/002_11', 'session1/002_12', 'session1/004_01',
#                    'session1/004_02', 'session1/004_03', 'session1/004_04', 'session1/004_05', 'session1/004_06',
#                    'session1/004_07', 'session1/004_08', 'session1/004_09', 'session1/004_10', 'session1/004_11',
#                    'session1/004_12', 'session1/007_01', 'session1/007_02', 'session1/007_03', 'session1/007_04',
#                    'session1/007_05', 'session1/007_06', 'session1/007_07', 'session1/007_08', 'session1/007_09',
#                    'session1/007_10', 'session1/007_11', 'session1/007_12', 'session1/008_01', 'session1/008_02',
#                    'session1/008_03', 'session1/008_04', 'session1/008_05', 'session1/008_06', 'session1/008_07',
#                    'session1/008_08', 'session1/008_09', 'session1/008_10', 'session1/008_11', 'session1/008_12',
#                    'session1/012_01', 'session1/012_02', 'session1/012_03', 'session1/012_04', 'session1/012_05',
#                    'session1/012_06', 'session1/012_07', 'session1/012_08', 'session1/012_09', 'session1/012_10',
#                    'session1/012_11', 'session1/012_12', 'session1/013_01', 'session1/013_02', 'session1/013_03',
#                    'session1/013_04', 'session1/013_05', 'session1/013_06', 'session1/013_07', 'session1/013_08',
#                    'session1/013_09', 'session1/013_10', 'session1/013_11', 'session1/013_12', 'session1/017_01',
#                    'session1/017_02', 'session1/017_03', 'session1/017_04', 'session1/017_05', 'session1/017_06',
#                    'session1/017_07', 'session1/017_08', 'session1/017_09', 'session1/017_10', 'session1/017_11',
#                    'session1/017_12', 'session1/003_01', 'session1/003_02', 'session1/003_03', 'session1/005_01',
#                    'session1/005_02', 'session1/005_03']
QUERY_SEQUENCES = ['session1/001_01', 'session1/001_02', 'session1/001_03', 'session1/001_04', 'session1/001_05',
                   'session1/001_06', 'session1/001_07', 'session1/001_08', 'session1/001_09', 'session1/001_10',
                   'session1/001_11', 'session1/001_12', 'session1/002_01', 'session1/002_02', 'session1/002_03',
                   'session1/002_04', 'session1/002_05', 'session1/002_06', 'session1/002_07', 'session1/002_08',
                   'session1/002_09', 'session1/002_10', 'session1/002_11', 'session1/002_12', 'session1/004_01',
                   'session1/004_02', 'session1/004_03', 'session1/004_04', 'session1/004_05', 'session1/004_06',
                   'session1/004_07', 'session1/004_08', 'session1/004_09', 'session1/004_10', 'session1/004_11',
                   'session1/004_12', 'session1/007_01', 'session1/007_02', 'session1/007_03', 'session1/007_04',
                   'session1/007_05', 'session1/007_06', 'session1/007_07', 'session1/007_08', 'session1/007_09',
                   'session1/007_10', 'session1/007_11', 'session1/007_12', 'session1/008_01', 'session1/008_02',
                   'session1/008_03', 'session1/008_04', 'session1/008_05', 'session1/008_06', 'session1/008_07',
                   'session1/008_08', 'session1/008_09', 'session1/008_10', 'session1/008_11', 'session1/008_12',
                   'session1/012_01', 'session1/012_02', 'session1/012_03', 'session1/012_04', 'session1/012_05',
                   'session1/012_06', 'session1/012_07', 'session1/012_08', 'session1/012_09', 'session1/012_10',
                   'session1/012_11', 'session1/012_12', 'session1/013_01', 'session1/013_02', 'session1/013_03',
                   'session1/013_04', 'session1/013_05', 'session1/013_06', 'session1/013_07', 'session1/013_08',
                   'session1/013_09', 'session1/013_10', 'session1/013_11', 'session1/013_12', 'session1/017_01',
                   'session1/017_02', 'session1/017_03', 'session1/017_04', 'session1/017_05', 'session1/017_06',
                   'session1/017_07', 'session1/017_08', 'session1/017_09', 'session1/017_10', 'session1/017_11',
                   'session1/017_12', 'session1/003_01', 'session1/003_02', 'session1/003_03', 'session1/005_01',
                   'session1/005_02', 'session1/005_03', 'session1/006_01', 'session1/006_02', 'session1/006_03',
                   'session1/009_01', 'session1/009_02', 'session1/009_03', 'session1/010_01', 'session1/010_02',
                   'session1/010_03', 'session1/011_01', 'session1/011_02', 'session1/011_03', 'session1/014_01',
                   'session1/014_02', 'session1/014_03', 'session1/015_01', 'session1/015_02', 'session1/015_03',
                   'session1/016_01', 'session1/016_02', 'session1/016_03', 'session1/018_01', 'session1/018_02',
                   'session1/018_03', 'session1/019_01', 'session1/019_02', 'session1/019_03', 'session1/020_01',
                   'session1/020_02', 'session1/020_03', 'session1/021_01', 'session1/021_02', 'session1/021_03',
                   'session1/022_01', 'session1/022_02', 'session1/022_03', 'session1/023_01', 'session1/023_02',
                   'session1/023_03', 'session1/024_01', 'session1/024_02', 'session1/024_03', 'session1/025_01',
                   'session1/025_02', 'session1/025_03', 'session1/026_01', 'session1/026_02', 'session1/026_03',
                   'session1/027_01', 'session1/027_02', 'session1/027_03', 'session1/028_01', 'session1/028_02',
                   'session1/028_03', 'session1/029_01', 'session1/029_02', 'session1/029_03', 'session1/030_01',
                   'session1/030_02', 'session1/030_03', 'session1/032_01', 'session1/032_02', 'session1/032_03',
                   'session1/033_01', 'session1/033_02', 'session1/033_03', 'session1/034_01', 'session1/034_02',
                   'session1/034_03', 'session1/035_01', 'session1/035_02', 'session1/035_03', 'session1/036_01',
                   'session1/036_02', 'session1/036_03', 'session1/037_01', 'session1/037_02', 'session1/037_03',
                   'session1/038_01', 'session1/038_02', 'session1/038_03', 'session1/039_01', 'session1/039_02',
                   'session1/039_03', 'session1/041_01', 'session1/041_02', 'session1/041_03', 'session1/042_01',
                   'session1/042_02', 'session1/042_03', 'session1/043_01', 'session1/043_02', 'session1/043_03',
                   'session1/044_01', 'session1/044_02', 'session1/044_03', 'session1/045_01', 'session1/045_02',
                   'session1/045_03', 'session1/046_01', 'session1/046_02', 'session1/046_03', 'session1/047_01',
                   'session1/047_02', 'session1/047_03', 'session1/049_01', 'session1/049_02', 'session1/049_03',
                   'session1/050_01', 'session1/050_02', 'session1/050_03', 'session1/051_01', 'session1/051_02',
                   'session1/051_03', 'session1/052_01', 'session1/052_02', 'session1/052_03', 'session1/053_01',
                   'session1/053_02', 'session1/053_03', 'session1/054_01', 'session1/054_02', 'session1/054_03',
                   'session1/055_01', 'session1/055_02', 'session1/055_03', 'session1/056_01', 'session1/056_02',
                   'session1/056_03', 'session1/057_01', 'session1/057_02', 'session1/057_03', 'session1/058_01',
                   'session1/058_02', 'session1/058_03', 'session1/059_01', 'session1/059_02', 'session1/059_03',
                   'session1/060_01', 'session1/060_02', 'session1/060_03', 'session1/061_01', 'session1/061_02',
                   'session1/061_03', 'session1/062_01', 'session1/062_02', 'session1/062_03', 'session1/063_01',
                   'session1/063_02', 'session1/063_03', 'session1/064_01', 'session1/064_02', 'session1/064_03',
                   'session1/065_01', 'session1/065_02', 'session1/065_03', 'session1/066_01', 'session1/066_02',
                   'session1/066_03', 'session1/067_01', 'session1/067_02', 'session1/067_03', 'session1/068_01',
                   'session1/068_02', 'session1/068_03', 'session1/069_01', 'session1/069_02', 'session1/069_03',
                   'session1/070_01', 'session1/070_02', 'session1/070_03', 'session1/071_01', 'session1/071_02',
                   'session1/071_03', 'session1/072_01', 'session1/072_02', 'session1/072_03', 'session1/073_01',
                   'session1/073_02', 'session1/073_03', 'session1/074_01', 'session1/074_02', 'session1/074_03',
                   'session1/075_01', 'session1/075_02', 'session1/075_03', 'session1/076_01', 'session1/076_02',
                   'session1/076_03', 'session1/078_01', 'session1/078_02', 'session1/078_03', 'session1/079_01',
                   'session1/079_02', 'session1/079_03', 'session1/080_01', 'session1/080_02', 'session1/080_03',
                   'session1/081_01', 'session1/081_02', 'session1/081_03', 'session1/082_01', 'session1/082_02',
                   'session1/082_03', 'session1/083_01', 'session1/083_02', 'session1/083_03', 'session1/084_01',
                   'session1/084_02', 'session1/084_03', 'session1/085_01', 'session1/085_02', 'session1/085_03',
                   'session1/086_01', 'session1/086_02', 'session1/086_03', 'session1/087_01', 'session1/087_02',
                   'session1/087_03', 'session1/088_01', 'session1/088_02', 'session1/088_03', 'session1/089_01',
                   'session1/089_02', 'session1/089_03', 'session1/090_01', 'session1/090_02', 'session1/090_03',
                   'session1/091_01', 'session1/091_02', 'session1/091_03', 'session1/092_01', 'session1/092_02',
                   'session1/092_03', 'session1/093_01', 'session1/093_02', 'session1/093_03', 'session1/094_01',
                   'session1/094_02', 'session1/094_03', 'session1/095_01', 'session1/095_02', 'session1/095_03',
                   'session1/096_01', 'session1/096_02', 'session1/096_03', 'session1/097_01', 'session1/097_02',
                   'session1/097_03', 'session1/098_01', 'session1/098_02', 'session1/098_03', 'session1/099_01',
                   'session1/099_02', 'session1/099_03', 'session1/100_01', 'session1/100_02', 'session1/100_03',
                   'session1/101_01', 'session1/101_02', 'session1/101_03', 'session1/102_01', 'session1/102_02',
                   'session1/102_03', 'session2/148_01', 'session2/148_02', 'session2/148_03', 'session2/149_01',
                   'session2/149_02', 'session2/149_03', 'session2/149_04', 'session2/149_05', 'session2/149_06',
                   'session2/149_10', 'session2/149_11', 'session2/149_12', 'session2/150_01', 'session2/150_02',
                   'session2/150_03', 'session2/150_04', 'session2/150_05', 'session2/150_06', 'session2/150_10',
                   'session2/150_11', 'session2/150_12', 'session2/151_01', 'session2/151_02', 'session2/151_03',
                   'session2/151_04', 'session2/151_05', 'session2/151_06', 'session2/151_10', 'session2/151_11',
                   'session2/151_12', 'session2/152_01', 'session2/152_02', 'session2/152_03', 'session2/152_04',
                   'session2/152_05', 'session2/152_06', 'session2/152_10', 'session2/152_11', 'session2/152_12',
                   'session2/153_01', 'session2/153_02', 'session2/153_03', 'session2/153_04', 'session2/153_05',
                   'session2/153_06', 'session2/154_01', 'session2/154_02', 'session2/154_03', 'session2/154_04',
                   'session2/154_05', 'session2/154_06', 'session2/154_10', 'session2/154_11', 'session2/154_12',
                   'session2/155_01', 'session2/155_02', 'session2/155_03', 'session2/155_04', 'session2/155_05',
                   'session2/155_06', 'session2/155_10', 'session2/155_11', 'session2/155_12', 'session2/156_01',
                   'session2/156_02', 'session2/156_03', 'session2/156_04', 'session2/156_05', 'session2/156_06',
                   'session2/156_10', 'session2/156_11', 'session2/156_12', 'session2/157_01', 'session2/157_02',
                   'session2/157_03', 'session2/157_04', 'session2/157_05', 'session2/157_06', 'session2/157_10',
                   'session2/157_11', 'session2/157_12', 'session2/158_01', 'session2/158_02', 'session2/158_03',
                   'session2/158_04', 'session2/158_05', 'session2/158_06', 'session2/158_10', 'session2/158_11',
                   'session2/158_12', 'session2/159_01', 'session2/159_02', 'session2/159_03', 'session2/159_04',
                   'session2/159_05', 'session2/159_06', 'session2/159_10', 'session2/159_11', 'session2/159_12',
                   'session2/160_01', 'session2/160_02', 'session2/160_03', 'session2/160_04', 'session2/160_05',
                   'session2/160_06', 'session2/160_10', 'session2/160_11', 'session2/160_12', 'session2/161_01',
                   'session2/161_02', 'session2/161_03', 'session2/161_04', 'session2/161_05', 'session2/161_06',
                   'session2/161_10', 'session2/161_11', 'session2/161_12', 'session2/162_01', 'session2/162_02',
                   'session2/162_03', 'session2/162_04', 'session2/162_05', 'session2/162_06', 'session2/162_10',
                   'session2/162_11', 'session2/162_12', 'session2/163_01', 'session2/163_02', 'session2/163_03',
                   'session2/163_04', 'session2/163_05', 'session2/163_06', 'session2/163_10', 'session2/163_11',
                   'session2/163_12', 'session2/164_01', 'session2/164_02', 'session2/164_03', 'session2/164_04',
                   'session2/164_05', 'session2/164_06', 'session2/164_10', 'session2/164_11', 'session2/164_12',
                   'session2/165_01', 'session2/165_02', 'session2/165_03', 'session2/165_04', 'session2/165_05',
                   'session2/165_06', 'session2/165_10', 'session2/165_11', 'session2/165_12', 'session2/166_01',
                   'session2/166_02', 'session2/166_03', 'session2/166_04', 'session2/166_05', 'session2/166_06',
                   'session2/166_10', 'session2/166_11', 'session2/166_12', 'session2/167_01', 'session2/167_02',
                   'session2/167_03', 'session2/167_04', 'session2/167_05', 'session2/167_06', 'session2/167_10',
                   'session2/167_11', 'session2/167_12', 'session2/168_01', 'session2/168_02', 'session2/168_03',
                   'session2/168_04', 'session2/168_05', 'session2/168_06', 'session2/168_10', 'session2/168_11',
                   'session2/168_12', 'session2/169_01', 'session2/169_02', 'session2/169_03', 'session2/169_04',
                   'session2/169_05', 'session2/169_06', 'session2/169_10', 'session2/169_11', 'session2/169_12',
                   'session2/170_01', 'session2/170_02', 'session2/170_03', 'session2/170_04', 'session2/170_05',
                   'session2/170_06', 'session2/170_10', 'session2/170_11', 'session2/170_12', 'session2/171_01',
                   'session2/171_02', 'session2/171_03', 'session2/171_04', 'session2/171_05', 'session2/171_06',
                   'session2/171_10', 'session2/171_11', 'session2/171_12', 'session2/172_01', 'session2/172_02',
                   'session2/172_03', 'session2/172_04', 'session2/172_05', 'session2/172_06', 'session2/172_10',
                   'session2/172_11', 'session2/172_12', 'session2/173_01', 'session2/173_02', 'session2/173_03',
                   'session2/173_04', 'session2/173_05', 'session2/173_06', 'session2/173_10', 'session2/173_11',
                   'session2/173_12', 'session2/174_01', 'session2/174_02', 'session2/174_03', 'session2/174_04',
                   'session2/174_05', 'session2/174_06', 'session2/174_10', 'session2/174_11', 'session2/174_12',
                   'session2/175_01', 'session2/175_02', 'session2/175_03', 'session2/175_04', 'session2/175_05',
                   'session2/175_06', 'session2/175_10', 'session2/175_11', 'session2/175_12', 'session2/176_01',
                   'session2/176_02', 'session2/176_03', 'session2/176_04', 'session2/176_05', 'session2/176_06',
                   'session2/176_10', 'session2/176_11', 'session2/176_12', 'session2/177_01', 'session2/177_02',
                   'session2/177_03', 'session2/177_04', 'session2/177_05', 'session2/177_06', 'session2/177_10',
                   'session2/177_11', 'session2/177_12', 'session2/178_01', 'session2/178_02', 'session2/178_03',
                   'session2/178_04', 'session2/178_05', 'session2/178_06', 'session2/178_10', 'session2/178_11',
                   'session2/178_12', 'session2/179_01', 'session2/179_02', 'session2/179_03', 'session2/179_04',
                   'session2/179_05', 'session2/179_06', 'session2/179_10', 'session2/179_11', 'session2/179_12',
                   'session2/180_01', 'session2/180_02', 'session2/180_03', 'session2/180_04', 'session2/180_05',
                   'session2/180_06', 'session2/180_10', 'session2/180_11', 'session2/180_12', 'session2/181_01',
                   'session2/181_02', 'session2/181_03', 'session2/181_04', 'session2/181_05', 'session2/181_06',
                   'session2/181_10', 'session2/181_11', 'session2/181_12', 'session2/182_01', 'session2/182_02',
                   'session2/182_03', 'session2/182_04', 'session2/182_05', 'session2/182_06', 'session2/182_10',
                   'session2/182_11', 'session2/182_12', 'session2/183_01', 'session2/183_02', 'session2/183_03',
                   'session2/183_04', 'session2/183_05', 'session2/183_06', 'session2/183_10', 'session2/183_11',
                   'session2/183_12', 'session2/184_01', 'session2/184_02', 'session2/184_03', 'session2/184_04',
                   'session2/184_05', 'session2/184_06', 'session2/184_10', 'session2/184_11', 'session2/184_12',
                   'session2/185_01', 'session2/185_02', 'session2/185_03', 'session2/185_04', 'session2/185_05',
                   'session2/185_06', 'session2/185_10', 'session2/185_11', 'session2/185_12', 'session2/186_01',
                   'session2/186_02', 'session2/186_03', 'session2/186_04', 'session2/186_05', 'session2/186_06',
                   'session2/186_10', 'session2/186_11', 'session2/186_12', 'session2/187_01', 'session2/187_02',
                   'session2/187_03', 'session2/187_04', 'session2/187_05', 'session2/187_06', 'session2/187_10',
                   'session2/187_11', 'session2/187_12', 'session2/188_01', 'session2/188_02', 'session2/188_03',
                   'session2/188_04', 'session2/188_05', 'session2/188_06', 'session2/188_10', 'session2/188_11',
                   'session2/188_12', 'session2/189_01', 'session2/189_02', 'session2/189_03', 'session2/189_04',
                   'session2/189_05', 'session2/189_06', 'session2/189_10', 'session2/189_11', 'session2/189_12',
                   'session2/190_01', 'session2/190_02', 'session2/190_03', 'session2/190_04', 'session2/190_05',
                   'session2/190_06', 'session2/190_10', 'session2/190_11', 'session2/190_12', 'session2/191_01',
                   'session2/191_02', 'session2/191_03', 'session2/191_04', 'session2/191_05', 'session2/191_06',
                   'session2/191_10', 'session2/191_11', 'session2/191_12', 'session2/192_01', 'session2/192_02',
                   'session2/192_03', 'session2/192_04', 'session2/192_05', 'session2/192_06', 'session2/192_10',
                   'session2/192_11', 'session2/192_12', 'session2/193_01', 'session2/193_02', 'session2/193_03',
                   'session2/193_04', 'session2/193_05', 'session2/193_06', 'session2/193_10', 'session2/193_11',
                   'session2/193_12', 'session2/194_01', 'session2/194_02', 'session2/194_03', 'session2/194_04',
                   'session2/194_05', 'session2/194_06', 'session2/194_10', 'session2/194_11', 'session2/194_12',
                   'session2/195_01', 'session2/195_02', 'session2/195_03', 'session2/195_04', 'session2/195_05',
                   'session2/195_06', 'session2/195_10', 'session2/195_11', 'session2/195_12', 'session2/196_01',
                   'session2/196_02', 'session2/196_03', 'session2/196_04', 'session2/196_05', 'session2/196_06',
                   'session2/196_10', 'session2/196_11', 'session2/196_12', 'session2/197_01', 'session2/197_02',
                   'session2/197_03', 'session2/197_04', 'session2/197_05', 'session2/197_06', 'session2/197_10',
                   'session2/197_11', 'session2/197_12', 'session2/198_01', 'session2/198_02', 'session2/198_03',
                   'session2/198_04', 'session2/198_05', 'session2/198_06', 'session2/198_10', 'session2/198_11',
                   'session2/198_12', 'session2/199_01', 'session2/199_02', 'session2/199_03', 'session2/199_04',
                   'session2/199_05', 'session2/199_06', 'session2/199_10', 'session2/199_11', 'session2/199_12',
                   'session2/200_01', 'session2/200_02', 'session2/200_03', 'session2/200_04', 'session2/200_05',
                   'session2/200_06', 'session2/200_10', 'session2/200_11', 'session2/200_12']

'''
Train sequences:
'''
TRAIN_SEQUENCES = ['session1/031_01', 'session1/031_02', 'session1/031_03', 'session1/031_04', 'session1/031_05',
                   'session1/031_06', 'session1/031_07', 'session1/031_08', 'session1/031_09', 'session1/031_10',
                   'session1/031_11', 'session1/031_12', 'session3/031_01', 'session3/031_02', 'session3/031_03',
                   'session3/031_04', 'session3/031_05', 'session3/031_06', 'session3/031_07', 'session3/031_08',
                   'session3/031_09', 'session3/031_10', 'session3/031_11', 'session3/031_12', 'session1/040_01',
                   'session1/040_02', 'session1/040_03', 'session1/040_04', 'session1/040_05', 'session1/040_06',
                   'session1/040_07', 'session1/040_08', 'session1/040_09', 'session1/040_10', 'session1/040_11',
                   'session1/040_12', 'session3/040_01', 'session3/040_02', 'session3/040_03', 'session3/040_04',
                   'session3/040_05', 'session3/040_06', 'session3/040_07', 'session3/040_08', 'session3/040_09',
                   'session3/040_10', 'session3/040_11', 'session3/040_12', 'session1/048_01', 'session1/048_02',
                   'session1/048_03', 'session1/048_04', 'session1/048_05', 'session1/048_06', 'session1/048_07',
                   'session1/048_08', 'session1/048_09', 'session1/048_10', 'session1/048_11', 'session1/048_12',
                   'session3/048_01', 'session3/048_02', 'session3/048_03', 'session3/048_04', 'session3/048_05',
                   'session3/048_06', 'session3/048_07', 'session3/048_08', 'session3/048_09', 'session3/048_10',
                   'session3/048_11', 'session3/048_12', 'session1/077_01', 'session1/077_02', 'session1/077_03',
                   'session1/077_04', 'session1/077_05', 'session1/077_06', 'session1/077_07', 'session1/077_08',
                   'session1/077_09', 'session1/077_10', 'session1/077_11', 'session1/077_12', 'session3/077_01',
                   'session3/077_02', 'session3/077_03', 'session3/077_04', 'session3/077_05', 'session3/077_06',
                   'session3/077_07', 'session3/077_08', 'session3/077_09', 'session3/077_10', 'session3/077_11',
                   'session3/077_12', 'session1/103_01', 'session1/103_02', 'session1/103_03', 'session1/103_04',
                   'session1/103_05', 'session1/103_06', 'session1/103_07', 'session1/103_08', 'session1/103_09',
                   'session1/103_10', 'session1/103_11', 'session1/103_12', 'session1/104_01', 'session1/104_02',
                   'session1/104_03', 'session1/104_04', 'session1/104_05', 'session1/104_06', 'session1/104_07',
                   'session1/104_08', 'session1/104_09', 'session1/104_10', 'session1/104_11', 'session1/104_12',
                   'session1/105_01', 'session1/105_02', 'session1/105_03', 'session1/105_04', 'session1/105_05',
                   'session1/105_06', 'session1/105_07', 'session1/105_08', 'session1/105_09', 'session1/105_10',
                   'session1/105_11', 'session1/105_12', 'session1/106_01', 'session1/106_02', 'session1/106_03',
                   'session1/106_04', 'session1/106_05', 'session1/106_06', 'session1/106_07', 'session1/106_08',
                   'session1/106_09', 'session1/106_10', 'session1/106_11', 'session1/106_12', 'session1/107_01',
                   'session1/107_02', 'session1/107_03', 'session1/107_04', 'session1/107_05', 'session1/107_06',
                   'session1/107_07', 'session1/107_08', 'session1/107_09', 'session1/107_10', 'session1/107_11',
                   'session1/107_12', 'session1/108_01', 'session1/108_02', 'session1/108_03', 'session1/108_04',
                   'session1/108_05', 'session1/108_06', 'session1/108_07', 'session1/108_08', 'session1/108_09',
                   'session1/108_10', 'session1/108_11', 'session1/108_12', 'session1/109_01', 'session1/109_02',
                   'session1/109_03', 'session1/109_04', 'session1/109_05', 'session1/109_06', 'session1/109_07',
                   'session1/109_08', 'session1/109_09', 'session1/109_10', 'session1/109_11', 'session1/109_12',
                   'session1/110_01', 'session1/110_02', 'session1/110_03', 'session1/110_04', 'session1/110_05',
                   'session1/110_06', 'session1/110_07', 'session1/110_08', 'session1/110_09', 'session1/110_10',
                   'session1/110_11', 'session1/110_12', 'session1/111_01', 'session1/111_02', 'session1/111_03',
                   'session1/111_04', 'session1/111_05', 'session1/111_06', 'session1/111_07', 'session1/111_08',
                   'session1/111_09', 'session1/111_10', 'session1/111_11', 'session1/111_12', 'session1/112_01',
                   'session1/112_02', 'session1/112_03', 'session1/112_04', 'session1/112_05', 'session1/112_06',
                   'session1/112_07', 'session1/112_08', 'session1/112_09', 'session1/112_10', 'session1/112_11',
                   'session1/112_12', 'session1/113_01', 'session1/113_02', 'session1/113_03', 'session1/113_04',
                   'session1/113_05', 'session1/113_06', 'session1/113_07', 'session1/113_08', 'session1/113_09',
                   'session1/113_10', 'session1/113_11', 'session1/113_12', 'session1/114_01', 'session1/114_02',
                   'session1/114_03', 'session1/114_04', 'session1/114_05', 'session1/114_06', 'session1/114_07',
                   'session1/114_08', 'session1/114_09', 'session1/114_10', 'session1/114_11', 'session1/114_12',
                   'session1/115_01', 'session1/115_02', 'session1/115_03', 'session1/115_04', 'session1/115_05',
                   'session1/115_06', 'session1/115_07', 'session1/115_08', 'session1/115_09', 'session1/115_10',
                   'session1/115_11', 'session1/115_12', 'session1/116_01', 'session1/116_02', 'session1/116_03',
                   'session1/116_04', 'session1/116_05', 'session1/116_06', 'session1/116_07', 'session1/116_08',
                   'session1/116_09', 'session1/116_10', 'session1/116_11', 'session1/116_12', 'session1/117_01',
                   'session1/117_02', 'session1/117_03', 'session1/117_04', 'session1/117_05', 'session1/117_06',
                   'session1/117_07', 'session1/117_08', 'session1/117_09', 'session1/117_10', 'session1/117_11',
                   'session1/117_12', 'session1/118_01', 'session1/118_02', 'session1/118_03', 'session1/118_04',
                   'session1/118_05', 'session1/118_06', 'session1/118_07', 'session1/118_08', 'session1/118_09',
                   'session1/118_10', 'session1/118_11', 'session1/118_12', 'session1/119_01', 'session1/119_02',
                   'session1/119_03', 'session1/119_04', 'session1/119_05', 'session1/119_06', 'session1/119_07',
                   'session1/119_08', 'session1/119_09', 'session1/119_10', 'session1/119_11', 'session1/119_12',
                   'session1/120_01', 'session1/120_02', 'session1/120_03', 'session1/120_04', 'session1/120_05',
                   'session1/120_06', 'session1/120_07', 'session1/120_08', 'session1/120_09', 'session1/120_10',
                   'session1/120_11', 'session1/120_12', 'session1/121_01', 'session1/121_02', 'session1/121_03',
                   'session1/121_04', 'session1/121_05', 'session1/121_06', 'session1/121_07', 'session1/121_08',
                   'session1/121_09', 'session1/121_10', 'session1/121_11', 'session1/121_12', 'session1/122_01',
                   'session1/122_02', 'session1/122_03', 'session1/122_04', 'session1/122_05', 'session1/122_06',
                   'session1/122_07', 'session1/122_08', 'session1/122_09', 'session1/122_10', 'session1/122_11',
                   'session1/122_12', 'session1/123_01', 'session1/123_02', 'session1/123_03', 'session1/123_04',
                   'session1/123_05', 'session1/123_06', 'session1/123_07', 'session1/123_08', 'session1/123_09',
                   'session1/123_10', 'session1/123_11', 'session1/123_12', 'session1/124_01', 'session1/124_02',
                   'session1/124_03', 'session1/124_04', 'session1/124_05', 'session1/124_06', 'session1/124_07',
                   'session1/124_08', 'session1/124_09', 'session1/124_10', 'session1/124_11', 'session1/124_12',
                   'session1/125_01', 'session1/125_02', 'session1/125_03', 'session1/125_04', 'session1/125_05',
                   'session1/125_06', 'session1/125_07', 'session1/125_08', 'session1/125_09', 'session1/125_10',
                   'session1/125_11', 'session1/125_12', 'session1/126_01', 'session1/126_02', 'session1/126_03',
                   'session1/126_04', 'session1/126_05', 'session1/126_06', 'session1/126_07', 'session1/126_08',
                   'session1/126_09', 'session1/126_10', 'session1/126_11', 'session1/126_12', 'session1/127_01',
                   'session1/127_02', 'session1/127_03', 'session1/127_04', 'session1/127_05', 'session1/127_06',
                   'session1/127_07', 'session1/127_08', 'session1/127_09', 'session1/127_10', 'session1/127_11',
                   'session1/127_12', 'session1/128_01', 'session1/128_02', 'session1/128_03', 'session1/128_04',
                   'session1/128_05', 'session1/128_06', 'session1/128_07', 'session1/128_08', 'session1/128_09',
                   'session1/128_10', 'session1/128_11', 'session1/128_12', 'session1/129_01', 'session1/129_02',
                   'session1/129_03', 'session1/129_04', 'session1/129_05', 'session1/129_06', 'session1/129_07',
                   'session1/129_08', 'session1/129_09', 'session1/129_10', 'session1/129_11', 'session1/129_12',
                   'session1/130_01', 'session1/130_02', 'session1/130_03', 'session1/130_04', 'session1/130_05',
                   'session1/130_06', 'session1/130_07', 'session1/130_08', 'session1/130_09', 'session1/130_10',
                   'session1/130_11', 'session1/130_12', 'session1/131_01', 'session1/131_02', 'session1/131_03',
                   'session1/131_04', 'session1/131_05', 'session1/131_06', 'session1/131_07', 'session1/131_08',
                   'session1/131_09', 'session1/131_10', 'session1/131_11', 'session1/131_12', 'session1/132_01',
                   'session1/132_02', 'session1/132_03', 'session1/132_04', 'session1/132_05', 'session1/132_06',
                   'session1/132_07', 'session1/132_08', 'session1/132_09', 'session1/132_10', 'session1/132_11',
                   'session1/132_12', 'session1/133_01', 'session1/133_02', 'session1/133_03', 'session1/133_04',
                   'session1/133_05', 'session1/133_06', 'session1/133_07', 'session1/133_08', 'session1/133_09',
                   'session1/133_10', 'session1/133_11', 'session1/133_12', 'session1/134_01', 'session1/134_02',
                   'session1/134_03', 'session1/134_04', 'session1/134_05', 'session1/134_06', 'session1/134_07',
                   'session1/134_08', 'session1/134_09', 'session1/134_10', 'session1/134_11', 'session1/134_12',
                   'session1/135_01', 'session1/135_02', 'session1/135_03', 'session1/135_04', 'session1/135_05',
                   'session1/135_06', 'session1/135_07', 'session1/135_08', 'session1/135_09', 'session1/135_10',
                   'session1/135_11', 'session1/135_12', 'session1/136_01', 'session1/136_02', 'session1/136_03',
                   'session1/136_04', 'session1/136_05', 'session1/136_06', 'session1/136_07', 'session1/136_08',
                   'session1/136_09', 'session1/136_10', 'session1/136_11', 'session1/136_12', 'session1/137_01',
                   'session1/137_02', 'session1/137_03', 'session1/137_04', 'session1/137_05', 'session1/137_06',
                   'session1/137_07', 'session1/137_08', 'session1/137_09', 'session1/137_10', 'session1/137_11',
                   'session1/137_12', 'session1/138_01', 'session1/138_02', 'session1/138_03', 'session1/138_04',
                   'session1/138_05', 'session1/138_06', 'session1/138_07', 'session1/138_08', 'session1/138_09',
                   'session1/138_10', 'session1/138_11', 'session1/138_12', 'session1/139_01', 'session1/139_02',
                   'session1/139_03', 'session1/139_04', 'session1/139_05', 'session1/139_06', 'session1/139_07',
                   'session1/139_08', 'session1/139_09', 'session1/139_10', 'session1/139_11', 'session1/139_12',
                   'session1/140_01', 'session1/140_02', 'session1/140_03', 'session1/140_04', 'session1/140_05',
                   'session1/140_06', 'session1/140_07', 'session1/140_08', 'session1/140_09', 'session1/140_10',
                   'session1/140_11', 'session1/140_12', 'session1/141_01', 'session1/141_02', 'session1/141_03',
                   'session1/141_04', 'session1/141_05', 'session1/141_06', 'session1/141_07', 'session1/141_08',
                   'session1/141_09', 'session1/141_10', 'session1/141_11', 'session1/141_12', 'session1/142_01',
                   'session1/142_02', 'session1/142_03', 'session1/142_04', 'session1/142_05', 'session1/142_06',
                   'session1/142_07', 'session1/142_08', 'session1/142_09', 'session1/142_10', 'session1/142_11',
                   'session1/142_12', 'session1/143_01', 'session1/143_02', 'session1/143_03', 'session1/143_04',
                   'session1/143_05', 'session1/143_06', 'session1/143_07', 'session1/143_08', 'session1/143_09',
                   'session1/143_10', 'session1/143_11', 'session1/143_12', 'session1/144_01', 'session1/144_02',
                   'session1/144_03', 'session1/144_04', 'session1/144_05', 'session1/144_06', 'session1/144_07',
                   'session1/144_08', 'session1/144_09', 'session1/144_10', 'session1/144_11', 'session1/144_12',
                   'session1/145_01', 'session1/145_02', 'session1/145_03', 'session1/145_04', 'session1/145_05',
                   'session1/145_06', 'session1/145_07', 'session1/145_08', 'session1/145_09', 'session1/145_10',
                   'session1/145_11', 'session1/145_12', 'session1/146_01', 'session1/146_02', 'session1/146_03',
                   'session1/146_04', 'session1/146_05', 'session1/146_06', 'session1/146_07', 'session1/146_08',
                   'session1/146_09', 'session1/146_10', 'session1/146_11', 'session1/146_12', 'session1/147_01',
                   'session1/147_02', 'session1/147_03', 'session1/147_04', 'session1/147_05', 'session1/147_06',
                   'session1/147_07', 'session1/147_08', 'session1/147_09', 'session1/147_10', 'session1/147_11',
                   'session1/147_12', 'session2/201_01', 'session2/201_02', 'session2/201_03', 'session2/201_04',
                   'session2/201_05', 'session2/201_06', 'session2/201_07', 'session2/201_08', 'session2/201_09',
                   'session2/201_10', 'session2/201_11', 'session2/201_12', 'session2/202_01', 'session2/202_02',
                   'session2/202_03', 'session2/202_04', 'session2/202_05', 'session2/202_06', 'session2/202_07',
                   'session2/202_08', 'session2/202_09', 'session2/202_10', 'session2/202_11', 'session2/202_12',
                   'session2/203_01', 'session2/203_02', 'session2/203_03', 'session2/203_04', 'session2/203_05',
                   'session2/203_06', 'session2/203_07', 'session2/203_08', 'session2/203_09', 'session2/203_10',
                   'session2/203_11', 'session2/203_12', 'session2/204_01', 'session2/204_02', 'session2/204_03',
                   'session2/204_04', 'session2/204_05', 'session2/204_06', 'session2/204_07', 'session2/204_08',
                   'session2/204_09', 'session2/204_10', 'session2/204_11', 'session2/204_12', 'session2/205_01',
                   'session2/205_02', 'session2/205_03', 'session2/205_04', 'session2/205_05', 'session2/205_06',
                   'session2/205_07', 'session2/205_08', 'session2/205_09', 'session2/205_10', 'session2/205_11',
                   'session2/205_12', 'session2/206_01', 'session2/206_02', 'session2/206_03', 'session2/206_04',
                   'session2/206_05', 'session2/206_06', 'session2/206_07', 'session2/206_08', 'session2/206_09',
                   'session2/206_10', 'session2/206_11', 'session2/206_12', 'session2/207_01', 'session2/207_02',
                   'session2/207_03', 'session2/207_04', 'session2/207_05', 'session2/207_06', 'session2/207_07',
                   'session2/207_08', 'session2/207_09', 'session2/207_10', 'session2/207_11', 'session2/207_12',
                   'session2/208_01', 'session2/208_02', 'session2/208_03', 'session2/208_04', 'session2/208_05',
                   'session2/208_06', 'session2/208_07', 'session2/208_08', 'session2/208_09', 'session2/208_10',
                   'session2/208_11', 'session2/208_12', 'session2/209_01', 'session2/209_02', 'session2/209_03',
                   'session2/209_04', 'session2/209_05', 'session2/209_06', 'session2/209_07', 'session2/209_08',
                   'session2/209_09', 'session2/209_10', 'session2/209_11', 'session2/209_12', 'session2/210_01',
                   'session2/210_02', 'session2/210_03', 'session2/210_04', 'session2/210_05', 'session2/210_06',
                   'session2/210_07', 'session2/210_08', 'session2/210_09', 'session2/210_10', 'session2/210_11',
                   'session2/210_12', 'session2/211_01', 'session2/211_02', 'session2/211_03', 'session2/211_04',
                   'session2/211_05', 'session2/211_06', 'session2/211_07', 'session2/211_08', 'session2/211_09',
                   'session2/211_10', 'session2/211_11', 'session2/211_12', 'session2/212_01', 'session2/212_02',
                   'session2/212_03', 'session2/212_04', 'session2/212_05', 'session2/212_06', 'session2/212_07',
                   'session2/212_08', 'session2/212_09', 'session2/212_10', 'session2/212_11', 'session2/212_12',
                   'session2/213_01', 'session2/213_02', 'session2/213_03', 'session2/213_04', 'session2/213_05',
                   'session2/213_06', 'session2/213_07', 'session2/213_08', 'session2/213_09', 'session2/213_10',
                   'session2/213_11', 'session2/213_12', 'session2/214_01', 'session2/214_02', 'session2/214_03',
                   'session2/214_04', 'session2/214_05', 'session2/214_06', 'session2/214_07', 'session2/214_08',
                   'session2/214_09', 'session2/214_10', 'session2/214_11', 'session2/214_12', 'session2/215_01',
                   'session2/215_02', 'session2/215_03', 'session2/215_04', 'session2/215_05', 'session2/215_06',
                   'session2/215_07', 'session2/215_08', 'session2/215_09', 'session2/215_10', 'session2/215_11',
                   'session2/215_12', 'session2/216_01', 'session2/216_02', 'session2/216_03', 'session2/216_04',
                   'session2/216_05', 'session2/216_06', 'session2/216_07', 'session2/216_08', 'session2/216_09',
                   'session2/216_10', 'session2/216_11', 'session2/216_12', 'session2/217_01', 'session2/217_02',
                   'session2/217_03', 'session2/217_04', 'session2/217_05', 'session2/217_06', 'session2/217_07',
                   'session2/217_08', 'session2/217_09', 'session2/217_10', 'session2/217_11', 'session2/217_12',
                   'session2/218_01', 'session2/218_02', 'session2/218_03', 'session2/218_04', 'session2/218_05',
                   'session2/218_06', 'session2/218_07', 'session2/218_08', 'session2/218_09', 'session2/218_10',
                   'session2/218_11', 'session2/218_12', 'session2/219_01', 'session2/219_02', 'session2/219_03',
                   'session2/219_04', 'session2/219_05', 'session2/219_06', 'session2/219_07', 'session2/219_08',
                   'session2/219_09', 'session2/219_10', 'session2/219_11', 'session2/219_12', 'session2/220_01',
                   'session2/220_02', 'session2/220_03', 'session2/220_04', 'session2/220_05', 'session2/220_06',
                   'session2/220_07', 'session2/220_08', 'session2/220_09', 'session2/220_10', 'session2/220_11',
                   'session2/220_12', 'session2/221_01', 'session2/221_02', 'session2/221_03', 'session2/221_04',
                   'session2/221_05', 'session2/221_06', 'session2/221_07', 'session2/221_08', 'session2/221_09',
                   'session2/221_10', 'session2/221_11', 'session2/221_12', 'session2/222_01', 'session2/222_02',
                   'session2/222_03', 'session2/222_04', 'session2/222_05', 'session2/222_06', 'session2/222_07',
                   'session2/222_08', 'session2/222_09', 'session2/222_10', 'session2/222_11', 'session2/222_12',
                   'session2/223_01', 'session2/223_02', 'session2/223_03', 'session2/223_04', 'session2/223_05',
                   'session2/223_06', 'session2/223_07', 'session2/223_08', 'session2/223_09', 'session2/223_10',
                   'session2/223_11', 'session2/223_12', 'session2/224_01', 'session2/224_02', 'session2/224_03',
                   'session2/224_04', 'session2/224_05', 'session2/224_06', 'session2/224_07', 'session2/224_08',
                   'session2/224_09', 'session2/224_10', 'session2/224_11', 'session2/224_12', 'session2/225_01',
                   'session2/225_02', 'session2/225_03', 'session2/225_04', 'session2/225_05', 'session2/225_06',
                   'session2/225_07', 'session2/225_08', 'session2/225_09', 'session2/225_10', 'session2/225_11',
                   'session2/225_12', 'session2/226_01', 'session2/226_02', 'session2/226_03', 'session2/226_04',
                   'session2/226_05', 'session2/226_06', 'session2/226_07', 'session2/226_08', 'session2/226_09',
                   'session2/226_10', 'session2/226_11', 'session2/226_12']

SUFFIXES = {
    'gallery': {'features_pkl': 'gallery_features.pkl', 'imgs_path': 'gallery_faces', 'folder_list': GALLERY_SEQUENCES},
    'query': {'features_pkl': 'query_features.pkl', 'imgs_path': 'query_faces', 'folder_list': QUERY_SEQUENCES,
              'cam_id': 2, 'f_digit': 1},
}


# faceDetector = FaceDetector(faces_data_path=None, thresholds=[0.8, 0.8, 0.8],
#                             keep_all=True, device='cuda:0', min_face_size=50)
arcface = ArcFace()


# imgs_folder = '/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID/session1/001_01'
# for im_path in os.listdir(imgs_folder):
#     img = cv2.imread(os.path.join(imgs_folder, im_path))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     ret, prob = faceDetector.facenet_detecor(img, return_prob=True)
#     plt.imshow(ret[0].permute(1, 2, 0)/255)
#     plt.show()


def write_face_crop(face_detector, full_image_path, face_output_path):
    # full image path without the suffix
    im = ins_get_image(full_image_path.split('.')[0])
    faces = face_detector.get(im)
    if len(faces) > 0 and faces[0]:
        bbox_face = faces[0]['bbox']
        X = int(bbox_face[0])
        Y = int(bbox_face[1])
        W = int(bbox_face[2])
        H = int(bbox_face[3])
        cropped_image = im[Y:H, X:W]
        cv2.imwrite(face_output_path, cropped_image)
        return True
    return False


def save_sequences_to_folder(data_type, folders=None, save_images_for_unsupervised_gallery=False):
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=1, det_thresh=0.8)
    os.makedirs(os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('imgs_path')), exist_ok=True)
    i = 0
    total_imgs = 0
    if save_images_for_unsupervised_gallery:
        unsupervised_gallery_path = os.path.join(OUR_CCVID_PATH, 'unsupervised_gallery_GT')
        os.makedirs(unsupervised_gallery_path, exist_ok=True)
        CCVID_to_fast_naming = {}

    print(f'Saving sequences for {data_type}')
    folders = SUFFIXES.get(data_type).get('folder_list')
    for folder in tqdm.tqdm(folders, total=len(folders)):
        try:
            imgs_folder = os.path.join(CCVID_DATA_PATH, folder)
            id = folder.split('/')[1].split('_')[0]  # format of folder is: 'sessionX/YYY_ZZ'. X is session number, YYY is the person id, ZZ is sequence number
            detected_faces_in_sequence = 0
            for im_path in list(reversed(sorted(os.listdir(imgs_folder)))):
                if detected_faces_in_sequence == 10:
                    break
                total_imgs += 1
                # img = cv2.imread(os.path.join(imgs_folder, im_path)}
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # ret, prob = faceDetector.facenet_detecor(img, return_prob=True)
                # if is_img(ret):
                output_im_name = os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('imgs_path'), f'{folder.replace("/", "_")}_{im_path}')
                write_face = write_face_crop(app, os.path.join(imgs_folder, im_path), output_im_name)
                if write_face:
                #     cv2.imwrite(os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('imgs_path'), f'{id}_{folder.replace("/", "_")}_{i:06d}.png'), np.array(ret[0].permute(1, 2, 0))[:, :, ::-1])
                    if save_images_for_unsupervised_gallery:
                        cam_id = SUFFIXES.get(data_type).get('cam_id')
                        f_digit = SUFFIXES.get(data_type).get('f_digit')
                        fast_im_name = f'{int(id):04d}_c{cam_id}_f{f_digit}{i:06d}.jpg'
                        shutil.copy(os.path.join(imgs_folder, im_path),
                                    os.path.join(unsupervised_gallery_path, fast_im_name))
                        CCVID_to_fast_naming[im_path] = fast_im_name
                    i += 1
                    detected_faces_in_sequence += 1
        except Exception as e:
            continue
    if save_images_for_unsupervised_gallery:
        pickle.dump(CCVID_to_fast_naming, open(os.path.join(OUR_CCVID_PATH, 'unsup_gallery_GT_CCVID_to_fast_map.pkl'), 'wb'))

    print(f'Detected faces in {i} out of {total_imgs}')


def create_feature_vectors(data_type, recreate=False):
    if os.path.isfile(os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('features_pkl'))) and not recreate:
        print(f'loading {data_type} features from pickle')
        img_names, features = pickle.load(open(os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('features_pkl')), 'rb'))
    else:
        print(f'creating {data_type} feature vectors')
        img_names = []
        features = []
        imgs_folder = os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('imgs_path'))
        for img_name in tqdm.tqdm(os.listdir(imgs_folder), total=len(os.listdir(imgs_folder))):
            img_names.append(img_name)
            features.extend([arcface.get_img_embedding_from_file(os.path.join(imgs_folder, img_name))])
        pickle.dump((img_names, features), open(os.path.join(OUR_CCVID_PATH, SUFFIXES.get(data_type).get('features_pkl')), 'wb'))
    return img_names, features


def evaluate_performance_face(th=0):
    q_feats, q_pids = create_feature_vectors('query')
    g_feats, g_pids = create_feature_vectors('gallery')
    total = 0
    correct = 0
    correct_scores = []
    incorrect_scores = []
    ids = set()
    query_to_gallery_sims = {}
    for i, query in tqdm.tqdm(enumerate(q_feats), total=len(q_feats)):
        gallery_sims = compute_similarity_to_gallery(query, g_feats)
        max_idx = np.argmax(gallery_sims)
        max_score = np.max(gallery_sims)
        if max_score < th:
            continue
        total += 1
        predicted_label = g_pids[max_idx]
        true_label = q_pids[i]
        query_to_gallery_sims['']
        if true_label == predicted_label:
            correct += 1
            correct_scores.append(max_score)
            ids.add(predicted_label)
        else:
            incorrect_scores.append(max_score)
    pickle.dump((correct_scores, incorrect_scores), open(os.path.join(OUR_CCVID_PATH, 'all_scores.pkl'), 'wb'))

    if total == 0:  # no image passed the threshold, add 1 to avoid devision by 0 (accuracy will be 0 anyway).
        total += 1
    print(f'Threshold: {th}, Total ids: {len(ids)} Num correct: {correct}, Total: {total}, Accuracy: {correct/total}')
    return len(ids), correct, len(incorrect_scores), correct/total


def find_CCVID_th():
    ths = np.arange(0, 1, 0.05)
    num_ids = []
    corrects = []
    incorrects = []
    accuracies = []
    for th in ths:
        print(f'Running with threshold: {th}')
        num_id, correct, incorrect, accuracy = evaluate_performance_face(th)
        num_ids.append(num_id)
        corrects.append(correct)
        incorrects.append(incorrect)
        accuracies.append(accuracy)

    df = pd.DataFrame({'Threshold': ths, 'Num Ids': num_ids, 'Correct': corrects, 'Incorrect': incorrects, 'Accuracy': accuracies})
    df.to_csv(CCVID_THRESHOLDS)


def compute_similarity_to_gallery(query_feature, gallery_features):
    sims = []
    for g_feat in gallery_features:
        sims.extend([arcface.model.compute_sim(query_feature, g_feat)])
    return sims


def FAST_create_unsupervised_gallery():
    GT_unsupervised_gallery_path = os.path.join(OUR_CCVID_PATH, 'GT_unsupervised_gallery')
    unsupervised_gallery_path = os.path.join(OUR_CCVID_PATH, 'unsupervised_gallery')
    os.makedirs(unsupervised_gallery_path, exist_ok=True)
    unsupervised_to_GT = {}
    g_feats, g_pids = create_feature_vectors('gallery')
    total_imgs = 0
    for im_path in tqdm.tqdm(os.listdir(GT_unsupervised_gallery_path), total=len(os.listdir(GT_unsupervised_gallery_path))):
        total_imgs += 1
        img = cv2.imread(os.path.join(GT_unsupervised_gallery_path, im_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret, prob = faceDetector.facenet_detecor(img, return_prob=True)
        if is_img(ret):
            face_img = np.array(ret[0].permute(1, 2, 0))[:, :, ::-1]
            face_img = cv2.resize(face_img, (112, 112))
            q_feat = arcface.model.get_feat(face_img)
            gallery_sims = compute_similarity_to_gallery(q_feat, g_feats)
            max_idx = np.argmax(gallery_sims)
            max_score = np.max(gallery_sims)
            if max_score < 0.5:
                continue
            predicted_label = g_pids[max_idx]
            new_im_name = f'{int(predicted_label):04d}_c{im_path.split("_c")[1]}'
            unsupervised_to_GT[new_im_name] = im_path
            shutil.copy(os.path.join(GT_unsupervised_gallery_path, im_path), os.path.join(unsupervised_gallery_path, new_im_name))
    pickle.dump(unsupervised_to_GT, open(os.path.join(OUR_CCVID_PATH, 'unsupervised_to_GT.pkl'), 'wb'))


def FAST_build_fast_reid_dataset(create_gallery=False, downsample_rate=1, create_orig_query=False, create_train=False):
    """
    The unsupervised gallery can be taken from
    /home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID_our_attempt/all_seq_0.8_0.8_0.8_min_50/unsupervised_gallery
    c1 - real gallery (gallery of CCVID)
    c2 - unsupervised gallery
    c3 - real query
    c4 - real training data
    """
    if create_gallery:
        orig_gallery = os.path.join(OUR_CCVID_FAST_DATASET, 'downsampled_gallery')
        os.makedirs(orig_gallery, exist_ok=True)
        gallery_total_imgs = 0
        for folder in tqdm.tqdm(GALLERY_SEQUENCES, total=len(GALLERY_SEQUENCES)):
            i = 0
            imgs_folder = os.path.join(CCVID_DATA_PATH, folder)
            id = folder.split('/')[1].split('_')[0]  # format of folder is: 'sessionX/YYY_ZZ'. X is session number, YYY is the person id, ZZ is sequence number
            for im_path in os.listdir(imgs_folder):
                if not i % downsample_rate:
                    shutil.copy(os.path.join(imgs_folder, im_path),
                                os.path.join(orig_gallery, f'{int(id):04d}_c1_f0{gallery_total_imgs:06d}.jpg'))
                    gallery_total_imgs += 1
                i += 1

    if create_orig_query:
        query_orig = os.path.join(OUR_CCVID_FAST_DATASET, 'query_orig')
        os.makedirs(query_orig, exist_ok=True)
        query_total_imgs = 0
        for folder in tqdm.tqdm(QUERY_SEQUENCES, total=len(QUERY_SEQUENCES)):
            imgs_folder = os.path.join(CCVID_DATA_PATH, folder)
            id = folder.split('/')[1].split('_')[0]  # format of folder is: 'sessionX/YYY_ZZ'. X is session number, YYY is the person id, ZZ is sequence number
            for im_path in os.listdir(imgs_folder):
                shutil.copy(os.path.join(imgs_folder, im_path),
                            os.path.join(query_orig, f'{int(id):04d}_c3_f1{query_total_imgs:06d}.jpg'))
                query_total_imgs += 1

    if create_train:
        train_orig = os.path.join(OUR_CCVID_FAST_DATASET, 'train_orig')
        os.makedirs(train_orig, exist_ok=True)
        train_total_imgs = 0
        for folder in tqdm.tqdm(TRAIN_SEQUENCES, total=len(TRAIN_SEQUENCES)):
            imgs_folder = os.path.join(CCVID_DATA_PATH, folder)
            id = folder.split('/')[1].split('_')[0]  # format of folder is: 'sessionX/YYY_ZZ'. X is session number, YYY is the person id, ZZ is sequence number
            for im_path in os.listdir(imgs_folder):
                shutil.copy(os.path.join(imgs_folder, im_path),
                            os.path.join(train_orig, f'{int(id):04d}_c4_f1{train_total_imgs:06d}.jpg'))
                train_total_imgs += 1


def FAST_query_with_unsup_ids():
    query_unsup = os.path.join(OUR_CCVID_FAST_DATASET, 'query_only_correct_unsup_ids')
    query_orig = os.path.join(OUR_CCVID_FAST_DATASET, 'query_orig')
    os.makedirs(query_unsup, exist_ok=True)
    unsup_imgs_path = os.listdir('/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID_our_attempt/fast_reid_only_unsup_ids/bounding_box_test')
    unsup_ids = np.unique([im.split('_')[0] for im in unsup_imgs_path])

    for query in os.listdir(query_orig):
        query_id = query.split('_')[0]
        if query_id in unsup_ids:
            shutil.copy(os.path.join(query_orig, query), query_unsup)


def FAST_create_unsup_gallery_only_correct():
    # Get the query to gallery mapping
    query_face_list = CCVID_create_query_faces_list()
    query_to_gallery = CCVID_predict_query_labels(query_face_list)

    unsup_gallery_only_correct = os.path.join(OUR_CCVID_FAST_DATASET, 'unsup_gallery_only_correct')
    os.makedirs(unsup_gallery_only_correct, exist_ok=True)
    total = 0
    for im_path, predicted_id in query_to_gallery.items():
        if predicted_id == im_path.split('/')[1].split('_')[0]:
            shutil.copy(os.path.join(CCVID_DATA_PATH, im_path),
                        os.path.join(unsup_gallery_only_correct, f'{int(predicted_id):04d}_c5_f1{total:06d}.jpg'))
            total += 1


def CCVID_create_query_faces_list(recreate=False):
    pickle_path = os.path.join(OUR_CCVID_PATH, 'query_face_list.pkl')
    if os.path.isfile(pickle_path) and not recreate:
        print('Getting query faces list from pickle')
        query_faces_list = pickle.load(open(pickle_path, 'rb'))
        return query_faces_list

    i = 0
    total_imgs = 0
    query_faces_list = []
    print(f'Detecting faces in query images')
    folders = QUERY_SEQUENCES
    for folder in tqdm.tqdm(folders, total=len(folders)):
        try:
            imgs_folder = os.path.join(CCVID_DATA_PATH, folder)
            for im_path in os.listdir(imgs_folder):
                total_imgs += 1
                img = cv2.imread(os.path.join(imgs_folder, im_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ret, prob = faceDetector.facenet_detecor(img, return_prob=True)
                if is_img(ret):
                    # if a face was detected add the path to the list:
                    query_faces_list.append(os.path.join(folder, im_path))
                    i += 1
        except Exception as e:
            continue
    print(f'Detected faces in {i} out of {total_imgs}')
    pickle.dump(query_faces_list, open(pickle_path, 'wb'))

    return query_faces_list


def CCVID_predict_query_labels(query_face_list, th, recreate=False):
    pickle_path = os.path.join(OUR_CCVID_PATH, f'predicted_query_th_{th}.pkl')
    if os.path.isfile(pickle_path) and not recreate:
        print('Getting query predictions from pickle')
        query_to_gallery = pickle.load(open(pickle_path, 'rb'))
        return query_to_gallery

    print(f'Predicting query identities with threshold: {th}')
    query_to_gallery = {}
    g_feats, g_pids = create_feature_vectors('gallery')
    total = 0
    for im_path in tqdm.tqdm(sorted(query_face_list), total=len(query_face_list)):
        img = cv2.imread(os.path.join(CCVID_DATA_PATH, im_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret, prob = faceDetector.facenet_detecor(img, return_prob=True)
        if is_img(ret):
            face_img = np.array(ret[0].permute(1, 2, 0))[:, :, ::-1]
            face_img = cv2.resize(face_img, (112, 112))
            q_feat = arcface.model.get_feat(face_img)
            gallery_sims = compute_similarity_to_gallery(q_feat, g_feats)
            max_idx = np.argmax(gallery_sims)
            max_score = np.max(gallery_sims)
            if max_score < th:
                continue
            total += 1
            predicted_label = g_pids[max_idx]
            query_to_gallery[im_path] = predicted_label
    print(f'Predicted identities for {total} query images.')
    pickle.dump(query_to_gallery, open(pickle_path, 'wb'))
    return query_to_gallery


def CCVID_map_query_to_gallery(query_to_gallery, th, recreate=False):
    pickle_path = os.path.join(OUR_CCVID_PATH, f'query_to_gallery_map_th_{th}.pkl')
    if os.path.isfile(pickle_path) and not recreate:
        print('Getting query to gallery map from pickle')
        query_to_gallery_mapping = pickle.load(open(pickle_path, 'rb'))
        return query_to_gallery_mapping

    print(f'Creating query to gallery map for threshold {th}')
    query_to_gallery_mapping = {}
    prev_session = ''
    prev_seq = ''
    cur_seq_votes = {}
    for img_path in sorted(query_to_gallery.keys()):
        cur_session, cur_seq = img_path.split('/')[:-1]
        predicted_id = query_to_gallery[img_path]
        # count all votes of the same sequence in a dictionary
        if prev_session == cur_session and prev_seq == cur_seq:
            if cur_seq_votes.get(predicted_id):
                cur_seq_votes[predicted_id] += 1
            else:
                cur_seq_votes[predicted_id] = 1

        # when done with a sequence, take the majority vote of its elements
        else:
            if not prev_session and not prev_seq:  # first image
                cur_seq_votes[query_to_gallery[img_path]] = 1
                prev_session = cur_session
                prev_seq = cur_seq
                continue
            query_to_gallery_mapping[f'{prev_session}/{prev_seq}'] = max(cur_seq_votes, key=cur_seq_votes.get)
            cur_seq_votes = {}
            cur_seq_votes[predicted_id] = 1
        prev_session = cur_session
        prev_seq = cur_seq

    query_to_gallery_mapping[f'{prev_session}/{prev_seq}'] = max(cur_seq_votes, key=cur_seq_votes.get)  # add last sequence
    pickle.dump(query_to_gallery_mapping, open(pickle_path, 'wb'))
    return query_to_gallery_mapping


def CCVID_evaluate_majority_vote(query_to_gallery, query_to_gallery_mapping):
    """
    Calculate hte accuracy for the face images after predicting the labeled based on the majority vote for the sequence.
    """
    correct = 0
    incorrect = 0
    for im_path in query_to_gallery.keys():
        parts = im_path.split('/')
        predicted_id = query_to_gallery_mapping[f'{parts[0]}/{parts[1]}']
        GT_id = parts[1].split('_')[0]
        if predicted_id == GT_id:
            correct += 1
        else:
            incorrect += 1
    print(f'Majority vote predicted correctly {correct} and wrong {incorrect}. Accuracy: {correct/(correct+incorrect)}')


def CCVID_copy_query_to_gallery(query_to_gallery_mapping, th):
    with open(os.path.join(CCVID_DATA_PATH, f'enriched_gallery_th_{th}.txt'), 'w') as f:
        same_seq_counter = {}
        for seq, predicted_id in query_to_gallery_mapping.items():
            if predicted_id in same_seq_counter:
                same_seq_counter[predicted_id] += 1
            else:
                same_seq_counter[predicted_id] = 1
            # if it is the first time making predictions (or if the gallery images changed) copy the sequences
            # copy_tree(os.path.join(CCVID_DATA_PATH, seq),
            #           os.path.join(CCVID_DATA_PATH, 'session4', f'{predicted_id}_{same_seq_counter[predicted_id]:02d}'))
            f.write(f'session4/{predicted_id}_{same_seq_counter[predicted_id]:02d} {predicted_id}\tu0_l0_s0_c0_a0\n')


def CCVID_copy_query_to_perfect_gallery(query_to_gallery_mapping):
    with open(os.path.join(CCVID_DATA_PATH, 'perfect_gallery.txt'), 'w') as f:
        same_seq_counter = {}
        for seq, predicted_id in query_to_gallery_mapping.items():
            GT_id = seq.split('/')[1].split('_')[0]
            if GT_id != predicted_id:
                continue
            if predicted_id in same_seq_counter:
                same_seq_counter[predicted_id] += 1
            else:
                same_seq_counter[predicted_id] = 1
            copy_tree(os.path.join(CCVID_DATA_PATH, seq),
                      os.path.join(CCVID_DATA_PATH, 'session5', f'{predicted_id}_{same_seq_counter[predicted_id]:02d}'))
            f.write(f'session5/{predicted_id}_{same_seq_counter[predicted_id]:02d} {predicted_id}\tu0_l0_s0_c0_a0\n')


def CCVID_prepare_dataset_with_unsup(th):
    query_face_list = CCVID_create_query_faces_list()
    query_to_gallery = CCVID_predict_query_labels(query_face_list, th=th, recreate=True)
    query_to_gallery_mapping = CCVID_map_query_to_gallery(query_to_gallery, th=th, recreate=True)
    # CCVID_evaluate_image_based_prediction(query_to_gallery_mapping)
    CCVID_evaluate_majority_vote(query_to_gallery, query_to_gallery_mapping)
    CCVID_copy_query_to_gallery(query_to_gallery_mapping, th)
    # CCVID_copy_query_to_perfect_gallery(query_to_gallery_mapping)


# save_sequences_to_folder('query', save_images_for_unsupervised_gallery=True)
# save_sequences_to_folder('gallery')
create_feature_vectors('query', recreate=True)
create_feature_vectors('gallery', recreate=True)

# evaluate_performance()
# create_unsupervised_gallery()

# unsupervised_to_GT = pickle.load(open(os.path.join(OUR_CCVID_PATH, 'unsupervised_to_GT.pkl'), 'rb'))
# build_fast_reid_dataset(create_gallery=False, create_orig_query=False, create_train=True)
# print('daniel')

# query_with_unsup_ids()

# CCVID_prepare_dataset_with_unsup()

# FAST_create_unsup_gallery_only_correct()
# FAST_query_with_unsup_ids()

# FAST_build_fast_reid_dataset(create_gallery=True, downsample_rate=10)
# score_th = pickle.load(open('/home/bar_cohen/raid/OUR_DATASETS/CCVID-Datasets/CCVID_our_attempt/all_seq_0.8_0.8_0.8_min_50/all_scores.pkl', 'rb'))

# print('herloo')
# find_CCVID_th()
# CCVID_prepare_dataset_with_unsup(th=0.55)

# save_sequences_to_folder('query', save_images_for_unsupervised_gallery=True)
