import pickle
import sys

import mmcv

import os
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


class GalleryCreator:
    def __init__(self, gallery_path, label_encoder,
                 min_face_size=50,
                 tracker_conf_threshold = 0.99,
                 device='cuda:1',
                 track_config=TRACKING_CONFIG_PATH,
                 track_checkpoint=TRACKING_CHECKPOINT,
                 ):
        self.tracking_model = init_model(track_config, track_checkpoint, device=device)
        self.tracker_conf_threshold = tracker_conf_threshold
        self.gallery_path = gallery_path
        self.global_video_counter = 0 # counts the unique videos entered for the gallery

        # if face_model == FACE_NET:
        self.faceDetector = FaceDetector(faces_data_path=None,thresholds=[0.98,0.98,0.98],
                                    keep_all=True,min_face_size=min_face_size,
                                    device=device)
        self.faceClassifer = FaceClassifer(num_classes=19, label_encoder=label_encoder, device=device)
        self.faceClassifer.model_ft.load_state_dict(torch.load(FACE_CHECKPOINT))
        self.faceClassifer.model_ft.eval()
        self.pose_estimator = PoseEstimator(pose_config=POSE_CONFIG, pose_checkpoint=POSE_CHECKPOINT, device='cuda:0') # TODO hard code configs here

    def add_video_to_gallery_using_FaceNet(self, video_path:str, skip_every=500):
        imgs = mmcv.VideoReader(video_path)
        self.global_video_counter += 1
        vid_name = video_path.split('/')[-1][9:-4]
        global_i = 0
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
                    face_bboxes, face_probs = self.faceDetector.facenet_detecor.detect(img=crop_im)
                    face_imgs = self.faceDetector.facenet_detecor.extract(crop_im, face_bboxes, save_path=None)
                    if face_imgs is not None and len(face_imgs) > 0:
                        face_img, face_prob = self.pose_estimator.find_matching_face(crop_im, face_bboxes, face_probs,
                                                                                    face_imgs)
                        if is_img(face_img):
                            face_img = normalize_image(face_img)
                            crop_candidates_faces.append(face_img)
                            crop_candidates_inds.append(i)
                except:
                    print("OMG WTF")
                    continue
            if len(crop_candidates_faces) > 0: # some faces where detected
                preds, outs = self.faceClassifer.predict(torch.stack(crop_candidates_faces))
                labels = [self.faceClassifer.le.inverse_transform([int(pred)])[0] for pred in preds]
                confidences = [out.argmax() for out in outs]
                crop_cands = [crop_im for i, crop_im in enumerate(crops_imgs) if i in crop_candidates_inds]
                for i , (crop_im, label, confidence) in enumerate(zip(crop_cands, labels, confidences)):
                    if confidence > 0.98:
                        print(label, confidence)
                        crop_name = f'{label:04d}_c{self.global_video_counter}_f{global_i:07d}.jpg'
                        # dir_path = os.path.join(self.gallery_path, ID_TO_NAME[label])
                        dir_path = self.gallery_path
                        os.makedirs(dir_path, exist_ok=True)
                        global_i += 1
                        mmcv.imwrite(np.array(crop_im), os.path.join(dir_path, crop_name))

def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    return result


if __name__ == '__main__':
    le = pickle.load(open("/mnt/raid1/home/bar_cohen/FaceData/le_19.pkl",'rb'))
    gc = GalleryCreator(gallery_path="/home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0808_no_skip_pose/bounding_box_test/",
                        label_encoder=le, device='cuda:1')
    gc.global_video_counter += 1


    gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/8.8.21_cam1/videos/IPCamera_20210808120339.avi", skip_every=1)
    gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/8.8.21_cam1/videos/IPCamera_20210808073000.avi", skip_every=1)
    gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/8.8.21_cam1/videos/IPCamera_20210808082440.avi", skip_every=1)
    gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/8.8.21_cam1/videos/IPCamera_20210808092457.avi", skip_every=1)
    #
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/30.7.21_cam1/videos/IPCamera_20210730101432.avi", skip_every=100)
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/30.7.21_cam1/videos/IPCamera_20210730111802.avi", skip_every=100)
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/30.7.21_cam1/videos/IPCamera_20210730072959.avi", skip_every=100)
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/30.7.21_cam1/videos/IPCamera_20210730085653.avi", skip_every=100)

    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/4.8.21_cam1/videos/IPCamera_20210804112702.avi", skip_every=1)
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/4.8.21_cam1/videos/IPCamera_20210804151703.avi", skip_every=1)
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/4.8.21_cam1/videos/IPCamera_20210804103053.avi", skip_every=1)
    # gc.add_video_to_gallery_using_FaceNet("/mnt/raid1/home/bar_cohen/Data-Shoham/4.8.21_cam1/videos/IPCamera_20210804072959.avi", skip_every=1)
