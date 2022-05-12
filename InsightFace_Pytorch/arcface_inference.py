import pickle
import tqdm
from model import Backbone
import torch
from torchvision import transforms as trans
import os
from PIL import Image
from mtcnn import MTCNN
import numpy as np

FACE_GALLERY = '/home/bar_cohen/raid/OUR_DATASETS/ArcFaceGallery/faces_clean'
FACE_GALLERY_PICKLE = '/home/bar_cohen/raid/OUR_DATASETS/ArcFaceGallery/pickles/face_gallery_embds.pkl'
ARCFACE_CKPT = '/home/bar_cohen/D-KinderGuardian/InsightFace_Pytorch/work_space/models/model_ir_se50.pth'


class ArcFace:
    def __init__(self, device='cuda:0', face_gallery_data=FACE_GALLERY, face_gallery_pkl=FACE_GALLERY_PICKLE, load_from_pkl=True):
        self.device = device
        self.g_feats = None
        self.g_ids = None
        self.face_gallery_data = face_gallery_data
        self.face_gallery_pkl = face_gallery_pkl
        self.model = None
        self.load_from_pkl = load_from_pkl
        self.mtcnn = MTCNN()

        # load the model:
        self.load_model()
        # create the gallery feature embeddings:
        self.get_gallery_embeddings(use_pickle=self.load_from_pkl)

    def load_model(self):
        model = Backbone(50, 0.6, 'ir_se').to(self.device)
        model.requires_grad_(False)
        cls_ckpt = ARCFACE_CKPT
        model.load_state_dict(torch.load(cls_ckpt, map_location=self.device))
        self.model = model.to(self.device)

    def get_gallery_embeddings(self, use_pickle=True):
        if use_pickle:
            print(f'Loading gallery embeddings from pickle at: {FACE_GALLERY_PICKLE}')
            gallery_embedding = pickle.load(open(FACE_GALLERY_PICKLE, 'rb'))
            self.g_feats = gallery_embedding.get('embeddings')
            self.g_ids = gallery_embedding.get('ids')
        embeddings = []
        ids = []
        imgs = os.listdir(self.face_gallery_data)
        print('Creating feature vectors for gallery images:')
        for im_path in tqdm.tqdm(imgs, total=len(imgs)):
            img = Image.open(os.path.join(self.face_gallery_data, im_path))
            embeddings.append(self.create_embedding(img))
            ids.append(im_path.split('_')[0])
        gallery_embedding = {'embeddings': embeddings, 'ids': ids}
        print(f'Saving gallery embeddings to pickle at: {FACE_GALLERY_PICKLE}')
        pickle.dump(gallery_embedding, open(FACE_GALLERY_PICKLE, 'wb'))
        self.g_feats = embeddings
        self.g_ids = ids

    def create_embedding(self, img):
        self.model.eval()
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        with torch.no_grad():
            embedding = self.model(transform(img).to(self.device).unsqueeze(0))
        return embedding

    def predict(self, imgs):
        """
        Given query images, create an embedding for every image and compute the distmat from the gallery features.
        """
        q_feats = []

        # create the features of all given images:
        for face_im in imgs:
            q_feats.append(self.create_embedding(face_im))

        # create the distmat of the q_feats and g_feats
        distmat = 1 - (np.array(q_feats) @ np.array(self.g_feats.T))

        # compute the minimal distance of every q_feat from the gallery
        best_match_in_gallery = np.argmin(distmat, axis=1)
        return self.g_ids[best_match_in_gallery], distmat

    def detect_face(self, img, thresholds=[0.8, 0.8, 0.8]):
        """
        Given an input image, detect the faces in it using the MTCNN model.
        """
        try:
            face_img = self.mtcnn.align(img, thresholds=thresholds)
            return face_img
        except:
            return None


def detect_faces_for_gallery(path, mtcnn):
    """
    Given a path to a folder with person crops, iterate over all images and save only the images that contain faces.
    """
    imgs = os.listdir(path)
    output_root = '/home/bar_cohen/raid/OUR_DATASETS/ArcFaceGallery'
    failed_imgs = 0
    failed_names = []
    for im_path in tqdm.tqdm(imgs, total=len(imgs)):
        img = Image.open(os.path.join(path, im_path))
        try:
            face_img, bbox = mtcnn.align(img, thresholds=[0.8, 0.8, 0.8])
        except:
            failed_imgs += 1
            failed_names.append(im_path)
            continue
        face_img.save(os.path.join(output_root, 'faces', im_path), quality=100, subsampling=0)  # these parameters preserve the original image quality
        img.save(os.path.join(output_root, 'crops_with_faces', im_path), quality=100, subsampling=0)  # these parameters preserve the original image quality
    print(f'Total number of images failed: {failed_imgs}')
    pickle.dump(failed_names, open(os.path.join(output_root, 'imgs_without_face.pkl'), 'wb'))


if __name__ == '__main__':
    pass
    # mtcnn = MTCNN()
    # detect_faces('/home/bar_cohen/raid/OUR_DATASETS/ArcFaceGallery/all_data_without_3007/data', mtcnn)
