import pickle
import sys

import mmcv

import os

sys.path.append('FaceDetection')

from FaceDetection.arcface import ArcFace, GALLERY_PKL_PATH, GPIDS_PKL_PATH, GALLERY_NO_UNKNOWNS, GPIDS_NO_UNKNOWNS

sys.path.append('mmpose')

import torch
import tqdm
import numpy as np

from FaceDetection.pose_estimator import PoseEstimator
from mmtracking.mmtrack.apis import init_model, inference_mot
from mmtrack.apis import inference_mot, init_model
from FaceDetection.augmentions import normalize_image
from FaceDetection.faceClassifer import FaceClassifer
from FaceDetection.faceDetector import FaceDetector, is_img

TRACKING_CONFIG_PATH = "/home/bar_cohen/KinderGuardian/mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py"
TRACKING_CHECKPOINT = "/home/bar_cohen/KinderGuardian/mmtracking/checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
FACE_CHECKPOINT = "/mnt/raid1/home/bar_cohen/FaceData/checkpoints/4.8 Val, 1.pth"
POSE_CONFIG = "/home/bar_cohen/D-KinderGuardian/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
POSE_CHECKPOINT =  "/home/bar_cohen/D-KinderGuardian/checkpoints/mmpose-hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
FACE_NET = 'FaceNet'
ARC_FACE = 'ArcFace'
GPATH = "/mnt/raid1/home/bar_cohen/42street/clusters_min_margin_low_threshold/"


class GalleryCreator:
    def __init__(self, gallery_path, label_encoder,
                 min_face_size=20,
                 tracker_conf_threshold = 0.99,
                 device='cuda:0',
                 cam_id = '1',
                 track_config=TRACKING_CONFIG_PATH,
                 track_checkpoint=TRACKING_CHECKPOINT,
                 create_in_fastreid_format=False,
                 ):
        self.tracking_model = init_model(track_config, track_checkpoint, device=device)
        self.tracker_conf_threshold = tracker_conf_threshold
        self.cam_id = cam_id
        if create_in_fastreid_format:
            os.makedirs(gallery_path)
            os.makedirs(os.path.join(gallery_path,'bounding_box_test'))
            os.makedirs(os.path.join(gallery_path,'bounding_box_train'))
            os.makedirs(os.path.join(gallery_path,'query'))
            self.gallery_path = os.path.join(gallery_path,'bounding_box_test')
        else:
            os.makedirs(gallery_path, exist_ok=True)
            self.gallery_path = gallery_path

        self.faceDetector = FaceDetector(faces_data_path=None,thresholds=[0.6, 0.7, 0.7],
                                    keep_all=True,min_face_size=min_face_size,
                                    device=device)

        self.arc = ArcFace(gallery_path=GPATH)
        self.arc.read_gallery_from_scratch()
        self.arc.save_gallery_to_pkl(GALLERY_PKL_PATH, GPIDS_PKL_PATH)
        self.arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH, gpid_path=GPIDS_PKL_PATH)
        self.pose_estimator = PoseEstimator(pose_config=POSE_CONFIG, pose_checkpoint=POSE_CHECKPOINT, device=device) # TODO hard code configs here
        self.global_i = 0


    def add_video_to_gallery(self, video_path:str, face_clf, skip_every=500):
        imgs = mmcv.VideoReader(video_path)
        high_tresh_face_detector = FaceDetector(faces_data_path=None, thresholds=[0.67, 0.67, 0.7],
                     keep_all=True, min_face_size=20,
                     device='cuda:0')
        vid_name = video_path.split('/')[-1][9:-4]
        for image_index, img in tqdm.tqdm(enumerate(imgs), total=len(imgs)):
            if image_index % skip_every != 0:
                continue
            result = tracking_inference(self.tracking_model, img, image_index,
                                        acc_threshold=float(self.tracker_conf_threshold))
            ids = list(map(int, result['track_results'][0][:, 0]))
            confs = result['track_results'][0][:, -1]
            crops_bboxes = result['track_results'][0][:, 1:-1]
            crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
            crop_candidates_inds = []
            crop_candidates_faces = []
            for i, (id, conf, crop_im) in enumerate(zip(ids, confs, crops_imgs)):
                try:
                    ## first detection run --- detect all faces in image
                    face_bboxes, face_probs = self.faceDetector.facenet_detecor.detect(img=crop_im)
                    face_imgs = self.faceDetector.facenet_detecor.extract(crop_im, face_bboxes, save_path=None)

                    # if only a single face exists, apply a high threshold to get a high resolution face image and body
                    if face_imgs is not None and len(face_imgs)  == 1:
                        face_bboxes, face_probs = high_tresh_face_detector.facenet_detecor.detect(img=crop_im)
                        face_imgs = high_tresh_face_detector.facenet_detecor.extract(crop_im, face_bboxes, save_path=None)
                        face_img, face_prob = self.pose_estimator.find_matching_face(crop_im, face_bboxes, face_probs,
                                                                                    face_imgs)
                        if is_img(face_img):
                            if face_clf == FACE_NET:
                                face_img = normalize_image(face_img)
                            crop_candidates_faces.append(face_img)
                            crop_candidates_inds.append(i)
                        # else: TODO write when pose estimation discards image

                except Exception as e:
                    # print(e)
                    # print('err')
                    continue
            crop_cands = [crop_im for i, crop_im in enumerate(crops_imgs) if i in crop_candidates_inds]
            if len(crop_candidates_faces) > 0: # some faces where detected
                for i, (face,crop_im) in enumerate(zip(crop_candidates_faces, crop_cands)):
                    numpy_img = face.permute(1, 2, 0).int().numpy().astype(np.uint8)
                    face = numpy_img[:, :, ::-1]
                    cur_score = self.arc.predict_img(face)
                    label = max(cur_score, key=cur_score.get)
                    # silly threshold
                    print(cur_score[label])
                    if cur_score[label] >= 0.25:
                        crop_name = f'{label:04d}_c{self.cam_id}_f{self.global_i:07d}.jpg'
                        # dir_path = os.path.join(self.gallery_path, ID_TO_NAME[label])
                        dir_path = self.gallery_path
                        os.makedirs(dir_path, exist_ok=True)
                        self.global_i += 1
                        mmcv.imwrite(np.array(crop_im), os.path.join(dir_path, crop_name))

    # Too lazy to refactor the code for several methods

def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    return result


if __name__ == '__main__':
    print("Thats right yall")
    le = pickle.load(open("/mnt/raid1/home/bar_cohen/FaceData/le_19.pkl",'rb'))
    gc = GalleryCreator(gallery_path="/mnt/raid1/home/bar_cohen/42street/part2_all_low_detector_threshold/", cam_id="5",
                        label_encoder=le, device='cuda:1', create_in_fastreid_format=True, tracker_conf_threshold=0.0)
    vid_path = "/mnt/raid1/home/bar_cohen/42street/val_videos_2/"
    vids = [os.path.join(vid_path, vid) for vid in os.listdir(vid_path)]
    # vids = ["/mnt/raid1/home/bar_cohen/42street/42street_tagged_vids/part3/part3_s22000_e22501.mp4"]
    for vid in vids:
        # uncomment if you want a gallery per video
        # vid_name = vid.split('/')[-1]
        # gc = GalleryCreator(
        #     gallery_path=f"/mnt/raid1/home/bar_cohen/42street/part1_galleries/{vid_name[5:]}/",
        #     label_encoder=le, device='cuda:0', create_in_fastreid_format=True)
        gc.add_video_to_gallery(vid,face_clf=ARC_FACE, skip_every=1)
        break
