#action constants
CLUSTER = 'cluster'
TRACK_AND_CROP = 'track_and_crop'

# crop and track hyper-params default
CROP_INDEX = 500 # cropes every 500 frames
ACC_THRESHOLD = 0.999
# NAME_TO_ID =  {
#     "Adam": 1,
#     "Avigail": 2,
#     "Ayelet": 3,
#     "Bar": 4,
#     "Batel": 5,
#     "Big-Gali": 6,
#     "Eitan": 7,
#     "Gali": 8,
#     "Guy": 9,
#     "Halel": 10,
#     "Lea": 11,
#     "Noga": 12,
#     "Ofir": 13,
#     "Omer": 14,
#     "Roni": 15,
#     "Sofi": 16,
#     "Sofi-Daughter": 17,
#     "Yahel": 18,
#     "Hagai": 19,
#     "Ella": 20,
#     "Daniel": 21
# }

NAME_TO_ID = {
    'Peggy Sawyer': 1,
    'Billy Lawer': 2,
    'Andy Lee' : 3,
    "Dylen" : 4,
    "Dorothy Brock" : 5,
    "Maggie Jones" : 6,
    "Pat Dennings" : 7,
    "Julian Marsh" : 8,
    "GuyDancer1" : 9,
    "Phyllis" : 10,
    "Friend": 11,
    "Unknown" : 12,
    "Blond": 13,
    }
ID_TO_NAME = {v:k for k,v in NAME_TO_ID.items()}

# clustering hyper params defaults
K_CLUSTERS = 10
Y_AXIS_NUM_CROPS = 10
X_AXIS_NUM_CROPS = 5
NUM_OF_CROPS_TO_VIEW = X_AXIS_NUM_CROPS * Y_AXIS_NUM_CROPS

SKIP_TRACK = 'skip'
APPROVE_TRACK = 'y'
DISCARD = 'discard'
SPLIT_TRACK = 's'
RELABEL = 'r'
VAGUE = 'v'
QUIT = 'q'

