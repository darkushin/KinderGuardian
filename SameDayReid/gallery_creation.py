import pickle

import mmcv
import os

import torch
import tqdm
import numpy as np
from DataProcessing.dataProcessingConstants import ID_TO_NAME, NAME_TO_ID
from FaceDetection.augmentions import normalize_image
from FaceDetection.faceClassifer import FaceClassifer
from FaceDetection.faceDetector import FaceDetector, is_img
from mmtracking.mmtrack.apis import init_model, inference_mot

TRACKING_CONFIG_PATH = "/home/bar_cohen/KinderGuardian/mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
TRACKING_CHECKPOINT = "/home/bar_cohen/KinderGuardian/mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"

class GalleryCreator:
    def __init__(self, gallery_path, label_encoder,
                 min_face_size=40,
                 tracker_conf_threshold = 0.98,
                 device='cuda:1',
                 track_config=TRACKING_CONFIG_PATH,
                 track_checkpoint=TRACKING_CHECKPOINT):
        self.gallery_path = gallery_path
        self.faceDetector = FaceDetector(faces_data_path=None,thresholds=[0.97,0.97,0.97],
                                    keep_all=False,min_face_size=min_face_size,
                                    device=device)
        self.faceClassfier = FaceClassifer(num_classes=21, label_encoder=label_encoder,device=device)
        self.tracking_model = init_model(track_config, track_checkpoint, device=device)
        self.tracker_conf_threshold = tracker_conf_threshold

    def add_video_to_gallery(self, video_path:str):
        imgs = mmcv.VideoReader(video_path)
        vid_name = video_path.split('/')[-1][9:-4]

        for image_index, img in tqdm.tqdm(enumerate(imgs), total=len(imgs)):
            result = tracking_inference(self.tracking_model, img, image_index,
                                        acc_threshold=float(self.tracker_conf_threshold))
            ids = list(map(int, result['track_results'][0][:, 0]))
            confs = result['track_results'][0][:, -1]
            crops_bboxes = result['track_results'][0][:, 1:-1]
            crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
            crop_candidates_inds = []
            crop_candidates_faces = []
            for i, (id, conf, crop_im) in enumerate(zip(ids, confs, crops_imgs)):
                face_img = self.faceDetector.facenet_detecor(crop_im)
                if is_img(face_img):
                    face_img = normalize_image(face_img)
                    crop_candidates_faces.append(face_img)
                    crop_candidates_inds.append(i)
            if len(crop_candidates_faces) > 0: # some faces where detected
                preds, _ = self.faceClassfier.predict(torch.stack(crop_candidates_faces))
                labels = [self.faceClassfier.le.inverse_transform([int(pred)])[0] for pred in preds]
                crop_cands = np.array(crops_imgs)[crop_candidates_inds]
                for i , (crop_im, label) in enumerate(zip(crop_cands, labels)):
                    crop_name = f'{label:04d}_c1_f{i:07d}_{vid_name}.jpg'
                    dir_path = os.path.join(self.gallery_path, ID_TO_NAME[label])
                    os.makedirs(dir_path, exist_ok=True)
                    mmcv.imwrite(np.array(crop_im), os.path.join(dir_path, crop_name))

def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    return result


def collect_faces_from_video(video_path:str) -> []:
    fd = FaceDetector(faces_data_path=None,thresholds=[0.97,0.97,0.97],keep_all=False,min_face_size=40)
    imgs = mmcv.VideoReader(video_path)
    ret = []
    for img in tqdm.tqdm(imgs):
        face_img, prob = fd.facenet_detecor(img, return_prob=True)
        if is_img(face_img):
            ret.append(face_img[0])
    return ret

def collect_faces_from_list_of_videos(list_of_videos:list):
    face_imgs = []
    for video_path in list_of_videos:
        face_imgs.extend(collect_faces_from_video(video_path=video_path))
    return face_imgs

if __name__ == '__main__':
    le = pickle.load(open("/mnt/raid1/home/bar_cohen/FaceData/le.pkl",'rb'))
    gc = GalleryCreator(gallery_path="/mnt/raid1/home/bar_cohen/OUR_DATASETS/same_day_gallery/",
                        label_encoder=le, device='cuda:1')
    gc.add_video_to_gallery("/mnt/raid1/home/bar_cohen/trimmed_videos/IPCamera_20210803105422/IPCamera_20210803105422_s0_e501.mp4")