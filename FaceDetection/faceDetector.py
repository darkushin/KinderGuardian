import os
import pickle
from collections import defaultdict

import PIL
import cv2
import seaborn as sns
import pandas as pd
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
    def __init__(self, raw_images_path:str=None, faces_data_path:str=None, thresholds=[0.8,0.8,0.8],
                 keep_all=False):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.raw_images_path = raw_images_path
        self.faces_data_path = faces_data_path
        self.keep_all = keep_all
        self.facenet_detecor = MTCNN(margin=40, select_largest=True, post_process=False, device=device,
                                     keep_all=self.keep_all, thresholds=thresholds)
        self.face_treshold = 0.90
        self.high_conf_face_imgs = defaultdict(list)

    def crop_top_third_and_sides(self, img):
        # pil_img = Image.fromarray(img)
        pil_img = img # TODO remove this !!!
        width, height = pil_img.size
        cropped_img = pil_img.crop((width*0.2, 0, width*0.8, height*0.3))
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
        ret = self.facenet_detecor(img, return_prob=False)
        self.facenet_detecor.keep_all = self.keep_all
        return ret

    def is_img(self, img):
        return img is not None and img is not img.numel()

    def get_single_face(self,img, return_prob):
        face_img = self.facenet_detecor(img, return_prob=return_prob)
        if self.is_img(face_img):
            if face_img.size()[0] > 1:  # two or more faces detected in the img crop
                # faceClassifer.imshow(face_img[0:2])
                face_img = self.crop_top_third_and_sides(img)
                face_img = self.detect_single_face(face_img)  # this returns a single img of dim 3
            else:
                face_img = face_img[0]  # current face_img shape is 1ximage size(dim=3), we only want the img itself
        return face_img


    def filter_out_non_face_corps(self) -> None:
        """ Given a set of image crop filter out all images without the faces present
        and update the high conf face img member """
        if self.faces_data_path:
            print("pickle path to images received, loading...")
            self.high_conf_face_imgs = pickle.load(open(self.faces_data_path,'rb'))
        else:
            # assert self.raw_images_path , 'Pickle to existing face images not found, please input raw images path'
            # raw_imgs_dict = read_labeled_croped_images(self.raw_images_path)
            face_crops = get_entries(filters={Crop.is_face == True,
                                              Crop.reviewed_one == True,
                                              Crop.is_vague == False,
                                              Crop.invalid == False}).all()
            crops_path = "/mnt/raid1/home/bar_cohen/"
            # raw_imgs_dict = {NAME_TO_ID[crop.label] : cv2.imread(os.path.join(crops_path, crop.vid_name, crop.im_name)) for crop in face_crops}
            raw_imgs_dict = defaultdict(list)
            for i, crop in enumerate(face_crops):
                print(i ,'/',len(face_crops))
                # fix for v, v_ issue
                name = crop.im_name
                if not os.path.isfile(os.path.join(crops_path, crop.vid_name, name)):
                    name = 'v'+crop.im_name[2:]
                img = Image.open(os.path.join(crops_path, crop.vid_name, name))
                raw_imgs_dict[NAME_TO_ID[crop.label]].append(img.copy())
                img.close()

                # if i > 100:
                #     break

            # y = [NAME_TO_ID[crop.label] for crop in face_crops]

            print('Number of unique ids', len(raw_imgs_dict.keys()))
            given_num_of_images = sum([len(raw_imgs_dict[i]) for i in raw_imgs_dict.keys()])
            print(f"Received a total of {given_num_of_images} images")
            counter = 0
            for id in raw_imgs_dict.keys():
                for img in raw_imgs_dict[id]:
                    print(f'{counter}/{given_num_of_images}')
                    self.facenet_detecor.keep_all = False
                    ret = self.get_single_face(img, False)
                    if self.is_img(ret):
                        self.high_conf_face_imgs[id].append(ret)
                    counter += 1
                # if counter > 100:
                #     break
                # if self.is_img(ret):
                    #     if len(ret) > 1: # two or more faces detected in image
                    #         top_third_img = self.crop_top_third_and_sides(img)
                    #         ret = self.facenet_detecor(top_third_img)
                    #         if ret > 1: # edge case- there are still multi images in top_third, take the most probable one
                    #             ret = ret[0]
                    #     self.high_conf_face_imgs[id].append(ret)
                    # if counter > 400:
                    #     break

            given_num_of_images_final = sum([len(self.high_conf_face_imgs[i]) for i in raw_imgs_dict.keys()])
            print(f'Post filter left with {given_num_of_images_final}')
            pickle.dump(self.high_conf_face_imgs, open('C:\KinderGuardian\FaceDetection\images_faces.pkl','wb'))

    def create_X_y_faces(self) -> ([] , []):
        """
        Iterate over high_conf_face_imgs and extract the X images and y labels
        """
        assert self.high_conf_face_imgs , "high conf face images must be non-empty"
        X, y = [], []
        for k in self.high_conf_face_imgs.keys():
            for img in self.high_conf_face_imgs[k]:
                X.append(img)
                y.append(k)
        return X, y

    def build_samples_hist(self, samples_dict:dict, title:str=None):
        """Create a Bar plot that represented the number of face samples from every id"""
        cnt = {ID_TO_NAME[int(k)] : [len(samples_dict[k])] for k in samples_dict.keys()}
        df = pd.DataFrame.from_dict(cnt)
        ax= sns.barplot(data=df)
        ax.set_xticklabels(df.columns, rotation=45)
        plt.title(title)
        plt.show()

def main():
    """Simple test of FaceDetector"""
    fd = FaceDetector(raw_images_path='/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned',
                      faces_data_path='C:\KinderGuardian\FaceDetection\imgs_with_face_highconf.pkl')
    fd.filter_out_non_face_corps()
    X,y = fd.create_X_y_faces()
    print(len(X), len(y))
    fd.build_samples_hist(fd.high_conf_face_imgs, 'Face Images Hist')
    # fd.build_samples_hist(read_labled_croped_images(fd.raw_images_path), title='Raw Images Hist')
