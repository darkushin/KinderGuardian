import insightface
import numpy as np
import cv2
import os

import torch
from DataProcessing.dataProcessingConstants import ID_TO_NAME
import tqdm

model_path = '/home/bar_cohen/D-KinderGuardian/insightface_test/checkpoints/w600k_r50.onnx'
IMG_SIZE = ((112,112))
# CHAR_2_ID = {'main':1, 'blond':2, 'guy2': 4, 'w1':5, 'w2':6, 'young_guy':7, 'mustache':8, 'young_w3':9}
# ID_2_NAME = {v:k for k, v in CHAR_2_ID.items()}


class ArcFace():

    def __init__(self,gallery_path=None):
        self.model = insightface.model_zoo.get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)  # given gpu id, if negative, then use cpu
        self.gallery_path = gallery_path
        self.gallery = None # call read_gallery to set
        self.gpids = None

    def get_img_embedding_from_file(self,file):
        img = cv2.imread(file)
        img = cv2.resize(img, IMG_SIZE)
        embedding = self.model.get_feat(img)
        return embedding

    def read_gallery(self):
        embeddings = []
        gpids = []
        folders = [os.path.join(self.gallery_path, file) for file in os.listdir(self.gallery_path)]
        for folder in folders:
            if os.path.isdir(folder):
                files = [os.path.join(self.gallery_path,folder, file) for file in os.listdir(os.path.join(self.gallery_path, folder))]
                embeddings.extend([self.get_img_embedding_from_file(file) for file in files])
                cur_id = os.path.join(self.gallery_path,folder).split('/')[-1]
                gpids.extend([cur_id] * len(files))

        self.gallery = np.array(embeddings)
        self.gpids = np.array(gpids)

    def create_feats(self, path):
        """
        Create feature vectors for all images in the given path.
        """
        embeddings = []
        files = [os.path.join(path, file) for file in os.listdir(path) if '.png' in file]
        embeddings.extend([self.get_img_embedding_from_file(file) for file in files])
        return np.array(embeddings)

    def img_tensor_to_cv2(self, face):
        numpy_img = face.permute(1, 2, 0).int().numpy().astype(np.uint8)
        face = numpy_img[:, :, ::-1]
        return face

    def predict_img(self, img):
        if type(img) == torch.Tensor:
            img = self.img_tensor_to_cv2(img)
        if img.shape != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)
        input_feat = self.model.get_feat(img)
        scores = {i:0 for i in ID_TO_NAME.keys()}
        for i in scores.keys():
            gallery_of_i = self.gallery[self.gpids == ID_TO_NAME[i]]
            if gallery_of_i is not None and len(gallery_of_i) > 0:
                cur_sims = [self.model.compute_sim(input_feat, cand) for cand in gallery_of_i]
                scores[i] = np.max(cur_sims)
        return scores

    def predict_track(self, imgs:np.array):
        scores = {i:0 for i in ID_TO_NAME.keys()}
        for img in imgs:
            cur_img_scores = self.predict_img(img)
            for k in scores.keys():
                scores[k] += cur_img_scores[k] / len(imgs)
        return scores

if __name__ == '__main__':
    gpath = "/mnt/raid1/home/bar_cohen/42street/clusters/"

    arc = ArcFace(gpath)
    img1_filepath = "/mnt/raid1/home/bar_cohen/42street/clusters/mustache/_s31000_e31501.mp4_75.png"
    img2_filepath = "/mnt/raid1/home/bar_cohen/42street/_s11000_e11501.mp4_200.png"

    img1 = cv2.imread(img1_filepath)
    img2 = cv2.imread(img2_filepath)
    both = np.array([img1,img2])

    # arc.predict(both)

    arc.read_gallery()
    # print(arc.gallery)
    arc.predict_track(both)









