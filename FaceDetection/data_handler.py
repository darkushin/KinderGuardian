import os
import pickle

import PIL
import numpy as np

import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn import preprocessing

from DataProcessing.dataProcessingConstants import ID_TO_NAME, NAME_TO_ID
from FaceDetection.augmentions import normalize_image
from FaceDetection.faceDetector import FaceDetector, is_img

TRAIN_DATES = ['0729', '0801', '0802','0804' ,'0805']
VAL_DATES = ['0803']
TEST_DATES = ['0730','0808']

def labelencode(label_encoder_output,X_train ,y_train, X_val ,y_val, X_test, y_test ,classes_to_drop:list):
    def drop_from_cur_set(X, y, le):
        indexs_to_drop = [index for index, cls in enumerate(y) if cls in classes_to_drop]
        X = [x for i, x in enumerate(X) if i not in indexs_to_drop]
        y = [cls for i, cls in enumerate(y) if i not in indexs_to_drop]
        y = le.transform(y)
        assert len(X) == len(y)
        return X,y

    """
    creates a label encoder based on inserted classes to drop
    Args:
        label_encoder_output: output path to save label encoder
        X_train ,X_val, X_test: the image tensors
        y_train, y_val, y_test: image labels
        classes_to_drop: label classes not to be trained upon

    Returns: X_train_transformed, y_train_transformed, .. , label-encoder

    """
    # creating an all y vector to account for all images
    y = [int(i) for cur_y_set in [y_train,y_val,y_test] for i in cur_y_set if i not in classes_to_drop]
    print(set(y))
    le = preprocessing.LabelEncoder()
    le.fit(y)
    print(le.classes_)
    X_train ,y_train = drop_from_cur_set(X_train,y_train, le)
    X_val ,y_val = drop_from_cur_set(X_val,y_val, le)
    X_test ,y_test = drop_from_cur_set(X_test,y_test , le)
    pickle.dump(le, open(os.path.join(label_encoder_output, 'le.pkl'), 'wb'))
    return X_train,y_train,X_val,y_val,X_test,y_test, le


def load_data(data_path):
    le = pickle.load(open(os.path.join(data_path, 'le.pkl'), 'rb'))
    dl_train = pickle.load(open(os.path.join(data_path, 'dl_train_1.pkl'), 'rb'))
    dl_val = pickle.load(open(os.path.join(data_path, 'df_val_1.pkl'), 'rb'))
    dl_test = pickle.load(open(os.path.join(data_path, 'df_test_1.pkl'), 'rb'))
    return le, dl_train, dl_val, dl_test


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, images:[], labels:[]):
        super(FacesDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        image = self.images[item]
        return image, label


def create_X_y_faces(high_conf_face_imgs:dict, save_images_path=''):
    def save(split_type):
        cur_path = os.path.join(save_images_path, split_type, str(ID_TO_NAME[k]))
        os.makedirs(cur_path, exist_ok=True)
        plt.imsave(os.path.join(cur_path, f'{crop.im_name}'), img_show)
    """
    Iterate over high_conf_face_imgs and extract the X images and y labels
    """
    assert high_conf_face_imgs , "high conf face images must be non-empty"
    X_train, X_val, X_test, y_train, y_val, y_test = [], [], [] , [], [], []
    counter = 0
    for k in high_conf_face_imgs.keys():
        for img,crop in high_conf_face_imgs[k]:
            date = str(crop.vid_name[4:8])
            img_show = img.permute(1, 2, 0).int().numpy().astype(np.uint8)
            if date in TEST_DATES:
                img = normalize_image(img)
                X_test.append(img)
                y_test.append(k)
                if save_images_path:
                    save('test')
            elif date in VAL_DATES:
                img = normalize_image(img)
                X_val.append(img)
                y_val.append(k)
                if save_images_path:
                    save('val')
            elif date in TRAIN_DATES:
                # WE PASS THE ORIGINAL IMAGE TO TRAIN, NORMALIZATION WILL BE DONE LATER
                X_train.append(img)
                y_train.append(k)
                if save_images_path:
                    save('train')
            else:
                raise Exception(f"Unrecognized Date received {date}")
            counter += 1

    return X_train ,y_train, X_val ,y_val, X_test, y_test

def load_old_data(data_path:str, reload_images=False):
    if not reload_images:
        x_train_add, y_train_add = pickle.load(open(os.path.join('/mnt/raid1/home/bar_cohen/FaceData/', 'old_face_data.pkl'), 'rb'))
    else:
        fd = FaceDetector(keep_all=False, thresholds=[0.97,0.97,0.97])
        x_train_add = []
        y_train_add = []
        for root, _ , files in os.walk(data_path):
            for file in tqdm.tqdm(files):
                img = PIL.Image.open(os.path.join(root, file))
                # ret, _ = fd.get_single_face(img, is_PIL_input=True,norm=False)
                ret = fd.facenet_detecor(img, return_prob=False)
                if is_img(ret):
                    label = int(file[0:4])
                    # ret = ret.permute(1, 2, 0).int().numpy().astype(np.uint8)
                    x_train_add.append(ret)
                    y_train_add.append(label)
                img.close()
        pickle.dump((x_train_add, y_train_add), open(os.path.join('/mnt/raid1/home/bar_cohen/FaceData/', 'old_face_data.pkl'), 'wb'))
    return x_train_add, y_train_add

