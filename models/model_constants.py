# model Runner actions constants
from DataProcessing import dataProcessingConstants
TRACKING = 'tracking'
RE_ID_EVAL = 're-id-eval'
RE_ID_TRAIN = 're-id-train'
REID_ACTIONS = [RE_ID_EVAL, RE_ID_TRAIN]
DATA_PROCESSING_ACTIONS = [dataProcessingConstants.CLUSTER, dataProcessingConstants.TRACK_AND_CROP]
CHOICES = [TRACKING] + REID_ACTIONS + DATA_PROCESSING_ACTIONS
