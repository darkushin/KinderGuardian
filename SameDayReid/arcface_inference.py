import sys
import pickle
from pprint import pprint

import tqdm
import torch
from torchvision import transforms as trans
import os
from PIL import Image
import numpy as np

from DataProcessing.DB.dal import get_entries, Crop
from DataProcessing.dataProcessingConstants import ID_TO_NAME

sys.path.append('/home/bar_cohen/KinderGuardian/InsightFace_Pytorch')
from InsightFace_Pytorch.model import Backbone
from InsightFace_Pytorch.mtcnn import MTCNN


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
        else:
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

        # q_feats_tensor = torch.Tensor(len(q_feats), 512)
        # torch.cat(q_feats, out=q_feats_tensor)
        # g_feats_tensor = torch.Tensor(len(self.g_feats), 512)
        # torch.cat(self.g_feats, out=g_feats_tensor)

        # q_feats_tensor = torch.stack(q_feats)
        # g_feats_tensor = torch.stack(self.g_feats)
        distmat = 1 - (torch.stack(q_feats).resize(len(q_feats), 512) @ torch.stack(self.g_feats).resize(len(self.g_feats),512).T)

        # distmat = 1 - (np.array(q_feats) @ np.array(self.g_feats.T))

        # compute the minimal distance of every q_feat from the gallery
        best_match_in_gallery = torch.argmin(distmat, dim=1)
        return np.array(self.g_ids)[best_match_in_gallery.cpu()], distmat

    def detect_face(self, img, thresholds=[0.8, 0.8, 0.8]):
        """
        Given an input image, detect the faces in it using the MTCNN model.
        """
        try:
            face_img = self.mtcnn.align(img, thresholds=thresholds)
            return face_img
        except:
            return None


def evaluate_with_DB():
    arc = ArcFace(device='cuda:1') # TODO something is wrong with cuda to, does not work with cuda:0
    correct = 0
    total_face_images = 0
    crops = get_entries(filters={Crop.reviewed_one == True,
                                      Crop.is_vague == False,
                                      Crop.invalid == False,
                                      }).all()
    # crops = [crop for crop in crops if str(crop.vid_name[4:8]) == '0730']
    print("Running Arc face Evaluation")
    crops_path = "/mnt/raid1/home/bar_cohen/"
    for i, crop in tqdm.tqdm(enumerate(crops), total=len(crops)):
        # print(i ,'/',len(face_crops))
        # fix for v, v_ issue
        name = crop.im_name
        if not os.path.isfile(os.path.join(crops_path, crop.vid_name, name)):
            name = 'v' + crop.im_name[2:]
        img = Image.open(os.path.join(crops_path, crop.vid_name, name))
        face_img = arc.detect_face(img)
        if face_img is not None:
            total_face_images += 1
            pred = ID_TO_NAME[int(arc.predict([face_img])[0])]
            if pred == crop.label:
                correct += 1
            else:
                pass
                # img.save(f"/mnt/raid1/home/bar_cohen/FaceData/temp/pred:{pred}_label{crop.label}_{i}.png", quality=100, subsampling=0)

    print(f'A total of {total_face_images}/{len(crops)} where extracted, with a total of {correct} of them classified correctly')
    print(f"The Total acc is {correct/total_face_images}%")




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
        img.close()
    print(f'Total number of images failed: {failed_imgs}')
    pickle.dump(failed_names, open(os.path.join(output_root, 'imgs_without_face.pkl'), 'wb'))


if __name__ == '__main__':
    evaluate_with_DB()
    # mtcnn = MTCNN()
    # detect_faces('/home/bar_cohen/raid/OUR_DATASETS/ArcFaceGallery/all_data_without_3007/data', mtcnn)
