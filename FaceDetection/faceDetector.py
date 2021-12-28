import pickle
from collections import defaultdict, Counter
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from DataProcessing.dataProcessingConstants import ID_TO_NAME
from DataProcessing.utils import read_labled_croped_images
from FaceDetection.facenet_pytorch import MTCNN, InceptionResnetV1
# from mtcnn.mtcnn import MTCNN as mtcnn_origin
from matplotlib import pyplot as plt
import torch
import torchvision

# import plotly.io as pio
# pio.renderers.default = "browser"

class FaceDetector():

    def __init__(self, raw_images_path:str=None, faces_data_path:str=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_images_path = raw_images_path
        self.faces_data_path = faces_data_path
        # self.mtcnn_detector = mtcnn_origin()
        self.facenet_detecor = MTCNN(margin=40, select_largest=False, post_process=False, device=device, thresholds=[0.8,0.8,0.8])
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.face_treshold = 0.90
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.high_conf_face_imgs = defaultdict(list)


    def filter_out_non_face_corps(self) -> None:
        """ Given a set of image crop filter out all images without the faces present"""
        if self.faces_data_path:
            print("pickle path to images received, loading...")
            self.high_conf_face_imgs = pickle.load(open(self.faces_data_path,'rb'))
        else:
            assert self.raw_images_path , 'Pickle to existing face images not found, please input raw images path'
            raw_imgs_dict = read_labled_croped_images(self.raw_images_path)

            print('Number of unique ids', len(raw_imgs_dict.keys()))
            given_num_of_images = sum([len(raw_imgs_dict[i]) for i in raw_imgs_dict.keys()])
            print(f"Received a total of {given_num_of_images} images")
            counter = 0
            for id in raw_imgs_dict.keys():
                for img in raw_imgs_dict[id]:
                    print(f'{counter}/{given_num_of_images}')
                    # plt.imshow(img)
                    # plt.show()
                    ret = self.facenet_detecor(img)
                    if ret is not None and ret is not ret.numel():
                        # print(ret.shape)
                        # plt.imshow(ret.permute(1, 2, 0).int().numpy())
                        # plt.show()
                        # input()
                        self.high_conf_face_imgs[id].append(ret)
                    counter += 1

            given_num_of_images_final = sum([len(self.high_conf_face_imgs[i]) for i in raw_imgs_dict.keys()])
            print(f'Post filter left with {given_num_of_images_final}')
            pickle.dump(self.high_conf_face_imgs, open('C:\KinderGuardian\FaceDetection\imgs_with_face_highconf.pkl','wb'))

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

if __name__ == '__main__':
    pass
    # C:\KinderGuardian\FaceDetection\imgs_with_face_highconf.pkl
    # fc = FaceDetector(raw_images_path='/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned', faces_data_path='C:\KinderGuardian\FaceDetection\imgs_with_face_highconf.pkl')
    # # read_labled_croped_images(self.raw_images_path)
    # fc.build_samples_hist(fc.high_conf_face_imgs, 'Face Images Hist')
    # fc.build_samples_hist(read_labled_croped_images(fc.raw_images_path), title='Raw Images Hist')
