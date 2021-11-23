# model Runner actions constants
from DataProcessing import dataProcessingConstants
TRACKING = 'tracking'
RE_ID = 're-id'

BASIC_ACTIONS = [TRACKING, RE_ID]
DATA_PROCESSING_ACTIONS = [dataProcessingConstants.CLUSTER, dataProcessingConstants.TRACK_AND_CROP]
CHOICES = BASIC_ACTIONS + DATA_PROCESSING_ACTIONS
