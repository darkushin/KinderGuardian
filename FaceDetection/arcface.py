import pickle

import insightface
import numpy as np
import cv2
import os

import torch
from DataProcessing.dataProcessingConstants import ID_TO_NAME
import tqdm
GALLERY_PKL_PATH = "/mnt/raid1/home/bar_cohen/42street/pkls/gallery.pkl"
GPIDS_PKL_PATH = "/mnt/raid1/home/bar_cohen/42street/pkls/gpids.pkl"
# GALLERY_PKL_PATH_MIN_FACE_MARGIN = "/mnt/raid1/home/bar_cohen/42street/pkls/gallery_mmargin.pkl"
# GPIDS_PKL_PATH_MIN_FACE_MARGIN = "/mnt/raid1/home/bar_cohen/42street/pkls/gpids_mmargin.pkl"
GPIDS_NO_UNKNOWNS = "/mnt/raid1/home/bar_cohen/42street/pkls/gpids_no_unknowns.pkl"
GALLERY_NO_UNKNOWNS = "/mnt/raid1/home/bar_cohen/42street/pkls/gallery_no_unknowns.pkl"

model_path = '/home/bar_cohen/D-KinderGuardian/insightface_test/checkpoints/w600k_r50.onnx'
IMG_SIZE = ((112,112))
# CHAR_2_ID = {'main':1, 'blond':2, 'guy2': 4, 'w1':5, 'w2':6, 'young_guy':7, 'mustache':8, 'young_w3':9}
# ID_2_NAME = {v:k for k, v in CHAR_2_ID.items()}


class ArcFace():

    def __init__(self,gallery_path=None, device=1):
        self.model = insightface.model_zoo.get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=device)  # given gpu id, if negative, then use cpu
        self.gallery_path = gallery_path
        self.gallery = None # call read_gallery to set
        self.gpids = None
        self.score_threshold = 0.3

    def get_img_embedding_from_file(self,file):
        img = cv2.imread(file)
        img = cv2.resize(img, IMG_SIZE)
        embedding = self.model.get_feat(img)
        return embedding

    def read_gallery_from_pkl(self, gallery_path:str, gpid_path:str):
        if os.path.isfile(gallery_path) and os.path.isfile(gpid_path):
            with open(gallery_path, 'rb') as f:
                self.gallery = pickle.load(f)
                f.close()

            with open(gpid_path, 'rb') as f:
                self.gpids = pickle.load(f)
                f.close()
        else:
            raise "Entered Gallery or GPID paths are incorrect."
        print("Done Loading Gallery from pkl paths")

    def read_gallery_from_scratch(self):
        print("Creating gallery from scratch...")
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

    def save_gallery_to_pkl(self, gallery_path:str , gpid_path:str):
        if self.gallery is not None and self.gpids is not None:
            with open(gallery_path, 'wb') as f:
                pickle.dump(self.gallery,f)
                f.close()

            with open(gpid_path, 'wb') as f:
                pickle.dump(self.gpids, f)
                f.close()

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
                mean_scores = np.mean(cur_sims)
                max_scores = np.max(cur_sims)
                top_5_mean_score = np.argpartition(cur_sims, 5)[-5:].mean()
                # print(ID_TO_NAME[i], "mean score:", mean_scores, "max score", max_scores)
                # scores[i] = mean_scores if mean_scores > self.score_threshold else 0
                # scores[i] = top_5_mean_score
                scores[i] = mean_scores if mean_scores > 0.25 else 0
                # if scores[i] > 0:
                    # print(f"Gotcha, {i}")
        return scores

    def predict_track(self, imgs:np.array):
        scores = {i:0 for i in ID_TO_NAME.keys()}
        for img in imgs:
            cur_img_scores = self.predict_img(img)
            for k in scores.keys():
                # taking the max score across the track
                scores[k] += cur_img_scores[k] / len(imgs) #TODO this was with a single image diviastion OMG
                # if cur_img_scores[k] > scores[k]:
                #     scores[k] = cur_img_scores[k]

        # normalize scores
        # np_ids_score = np.array(list(scores.values()))
        # mean_score = np.mean(np_ids_score)
        # std_score = np.std(np_ids_score)
        #
        # # max_score = max(scores.values())
        # # if max_score > 0:
        # for k, v in scores.items():
        #     scores[k] = (v - mean_score) / (std_score + 0.000001)

        return scores

if __name__ == '__main__':
    gpath = "/mnt/raid1/home/bar_cohen/42street/corrected_face_clusters/"

    # arc = ArcFace(gpath)
    # img1_filepath = "/mnt/raid1/home/bar_cohen/42street/clusters/mustache/_s31000_e31501.mp4_75.png"
    # img2_filepath = "/mnt/raid1/home/bar_cohen/42street/_s11000_e11501.mp4_200.png"
    #
    # img1 = cv2.imread(img1_filepath)
    # img2 = cv2.imread(img2_filepath)
    # both = np.array([img1,img2])
    # # arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH_MIN_FACE_MARGIN, gpid_path=GPIDS_PKL_PATH_MIN_FACE_MARGIN)
    # # arc.model.forward(both)
    # # arc.predict(both)
    #
    # arc.read_gallery_from_scratch()
    # arc.save_gallery_to_pkl(GALLERY_PKL_PATH_MIN_FACE_MARGIN, GPIDS_PKL_PATH_MIN_FACE_MARGIN)
    # print(arc.gallery)
    # arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH, gpid_path=GPIDS_PKL_PATH)










