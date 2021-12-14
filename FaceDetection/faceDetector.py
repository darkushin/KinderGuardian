import pickle
from collections import defaultdict

from PIL import Image
from sklearn.model_selection import train_test_split

from DataProcessing.utils import read_labled_croped_images
from FaceDetection.facenet_pytorch import MTCNN, InceptionResnetV1
from mtcnn.mtcnn import MTCNN as mtcnn_origin
from matplotlib import pyplot as plt
import torch

class FaceDetector():

    def __init__(self):
        self.raw_imgs_dict = read_labled_croped_images('/home/bar_cohen/Data-Shoham/Labeled-Data-Cleaned')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mtcnn_detector = mtcnn_origin()
        self.facenet_detecor = MTCNN(margin=40, select_largest=False, post_process=False, device=device, thresholds=[0.8,0.8,0.8])
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.high_conf_face_imgs = defaultdict(list)
        self.face_treshold = 0.90
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def filter_out_non_face_corps(self, pickle_path=None) -> None:
        """ Given a set of image crop filter out all images without the faces present"""
        if pickle_path:
            print("pickle path to images received, loading...")
            self.high_conf_face_imgs = pickle.load(open(pickle_path,'rb'))
        else:

            given_num_of_images = sum([len(self.raw_imgs_dict[i]) for i in self.raw_imgs_dict.keys()])
            print(f"Received a total of {given_num_of_images} images")
            counter = 0
            for id in self.raw_imgs_dict.keys():
                for img in self.raw_imgs_dict[id]:
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
            given_num_of_images_final = sum([len(self.high_conf_face_imgs[i]) for i in self.raw_imgs_dict.keys()])
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


    def build_basic_face_classifier(self):
        X, y = self.create_X_y_faces()
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1 , random_state = 1)
        del X, y
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
        X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)

        self.resnet


















if __name__ == '__main__':
    fc = FaceClassifer()
    fc.filter_out_non_face_corps()

    

