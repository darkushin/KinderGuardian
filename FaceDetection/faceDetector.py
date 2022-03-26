import os
import pickle
from collections import defaultdict

import PIL
import cv2
import mmcv
import seaborn as sns
import pandas as pd
import tqdm
from PIL import Image
from torchvision import transforms

from DataProcessing.DB.dal import get_entries, Crop
from DataProcessing.dataProcessingConstants import ID_TO_NAME, NAME_TO_ID
from FaceDetection.facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
import torch

class FaceDetector():
    """
    The faceDetector class is in charge of using the MTCNN face detector
    params:
    raw_images_path - path to folder containing images for face detection
    face_data_path - path to folder already containing face images. This will be used in order to put said images into
    the correct format.
    """
    def __init__(self, faces_data_path:str=None, thresholds=[0.8,0.8,0.8],
                 keep_all=False, device='cuda:1'):
        self.device = device
        self.faces_data_path = faces_data_path
        self.keep_all = keep_all
        self.facenet_detecor = MTCNN(margin=40, select_largest=True, post_process=False, device=device,
                                     keep_all=self.keep_all, thresholds=thresholds)
        self.high_conf_face_imgs = defaultdict(list)

    def crop_top_third_and_sides(self, img, is_PIL_input):
        # this assumes img is a PIL obj
        if not is_PIL_input:
            img = PIL.Image.fromarray(img)
        width, height = img.size
        cropped_img = img.crop((width*0.2, 0, width*0.8, height*0.3))
        return cropped_img

    def detect_single_face_inv_norm(self, img):
        # use face detector to find a single face in an image, rests to entered init of keeping face after change
        self.facenet_detecor.keep_all = False
        comp = transforms.Compose([
            transforms.ToTensor(),
            lambda x:x*255,
        ])
        new_im = comp(img).permute(1,2,0).int()
        ret, prob = self.facenet_detecor(new_im, return_prob=True)
        self.facenet_detecor.keep_all = self.keep_all
        return ret , prob

    def detect_single_face(self, img):
        # use face detector to find a single face in an image, rests to entered init of keeping face after change
        self.facenet_detecor.keep_all = False
        ret, prob = self.facenet_detecor(img, return_prob=True)
        self.facenet_detecor.keep_all = self.keep_all
        return ret , prob

    def is_img(self, img):
        return img is not None and img is not img.numel()

    def get_single_face(self,img, is_PIL_input):
        face_img, prob = self.facenet_detecor(img, return_prob=True)
        if self.is_img(face_img):
            if face_img.size()[0] > 1:  # two or more faces detected in the img crop
                # faceClassifer.imshow(face_img[0:2])
                face_img = self.crop_top_third_and_sides(img, is_PIL_input)
                if is_PIL_input:
                    face_img, prob = self.detect_single_face(face_img)  # this returns a single img of dim 3
                else:
                    face_img, prob = self.detect_single_face_inv_norm(face_img)
            else:
                face_img = face_img[0]  # current face_img shape is 1ximage size(dim=3), we only want the img itself
                prob = prob[0]
        return face_img , prob


    def filter_out_non_face_corps(self) -> None:
        """ Given a set of image crop filter out all images without the faces present
        and update the high conf face img member """
        if self.faces_data_path:
            print("pickle path to images received, loading...")
            self.high_conf_face_imgs = pickle.load(open(os.path.join(self.faces_data_path, 'images_with_crop.pkl'),'rb'))
        else:
            face_crops = get_entries(filters={Crop.is_face == True,
                                              Crop.reviewed_one == True,
                                              Crop.is_vague == False,
                                              Crop.invalid == False}).all()
            crops_path = "/mnt/raid1/home/bar_cohen/"
            raw_imgs_dict = defaultdict(list)
            for i, crop in enumerate(face_crops):
                print(i ,'/',len(face_crops))
                # fix for v, v_ issue
                name = crop.im_name
                if not os.path.isfile(os.path.join(crops_path, crop.vid_name, name)):
                    name = 'v'+crop.im_name[2:]
                img = Image.open(os.path.join(crops_path, crop.vid_name, name))
                raw_imgs_dict[NAME_TO_ID[crop.label]].append((img.copy(),crop))
                img.close()

            print('Number of unique ids', len(raw_imgs_dict.keys()))
            given_num_of_images = sum([len(raw_imgs_dict[i]) for i in raw_imgs_dict.keys()])
            print(f"Received a total of {given_num_of_images} images")
            counter = 0
            for id in raw_imgs_dict.keys():
                for img,crop in raw_imgs_dict[id]:
                    print(f'{counter}/{given_num_of_images}')
                    ret, _ = self.get_single_face(img, is_PIL_input=True)
                    if self.is_img(ret):
                        print(crop.vid_name[4:8])
                        self.high_conf_face_imgs[id].append((ret,crop))
                    counter += 1

            given_num_of_images_final = sum([len(self.high_conf_face_imgs[i]) for i in raw_imgs_dict.keys()])
            print(f'Post filter left with {given_num_of_images_final}')
            pickle.dump(self.high_conf_face_imgs, open(os.path.join('/mnt/raid1/home/bar_cohen/FaceData/', 'images_with_crop.pkl'),'wb'))


    def create_X_y_faces(self) -> ([] , []):
        """
        Iterate over high_conf_face_imgs and extract the X images and y labels
        """
        assert self.high_conf_face_imgs , "high conf face images must be non-empty"
        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [] , [], [], []
        for k in self.high_conf_face_imgs.keys():
            for img,crop in self.high_conf_face_imgs[k]:
                date = crop.vid_name[4:8]
                print(date)
                if date in ['3007', '0808']:
                    X_test.append(img)
                    y_test.append(k)
                elif date in ['0802','0803']:
                    X_val.append(img)
                    y_val.append(k)
                else:
                    X_train.append(img)
                    y_train.append(k)
        return X_train ,y_train, X_val ,y_val, X_test, y_test

    def build_samples_hist(self, samples_dict:dict, title:str=None):
        """Create a Bar plot that represented the number of face samples from every id"""
        cnt = {ID_TO_NAME[int(k)] : [len(samples_dict[k])] for k in samples_dict.keys()}
        df = pd.DataFrame.from_dict(cnt)
        ax= sns.barplot(data=df)
        ax.set_xticklabels(df.columns, rotation=45)
        plt.title(title)
        plt.show()


def collect_faces_from_video(video_path:str) -> []:
    fd = FaceDetector(faces_data_path=None,thresholds=[0.97,0.97,0.97],keep_all=True)
    imgs = mmcv.VideoReader(video_path)
    ret = []
    for img in tqdm.tqdm(imgs):
        face_img, prob = fd.facenet_detecor(img, return_prob=True)
        if fd.is_img(face_img):
            if face_img.size()[0] == 1:  # two or more faces detected in the img crop
                ret.append(face_img[0])
    return ret

def collect_faces_from_list_of_videos(list_of_videos:list):
    face_imgs = []
    for video_path in list_of_videos:
        face_imgs.extend(collect_faces_from_video(video_path=video_path))
    return face_imgs

def main():
    """Simple test of FaceDetector"""
    fd = FaceDetector(raw_images_path='/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned',
                      faces_data_path='C:\KinderGuardian\FaceDetection\imgs_with_face_highconf.pkl')
    fd.filter_out_non_face_corps()
    X,y, _, _,_,_ = fd.create_X_y_faces() # only taining
    print(len(X), len(y))
    fd.build_samples_hist(fd.high_conf_face_imgs, 'Face Images Hist')
    # fd.build_samples_hist(read_labled_croped_images(fd.raw_images_path), title='Raw Images Hist')
