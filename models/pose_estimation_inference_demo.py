import os
import sys
import numpy as np

sys.path.append('mmpose')

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo
from FaceDetection.faceDetector import FaceDetector, is_img
from PIL import Image, ImageDraw

KEYPOINTS_TO_CHECK = ['nose', 'left_eye', 'right_eye']  # , 'left_ear', 'right_ear']


def does_face_match_to_pose(pose_results, face_bbox):
    does_match = True
    for keypoint in KEYPOINTS_TO_CHECK:
        keypoint_x, keypoint_y, conf = pose_results[0]['keypoints'][dataset_info.keypoint_name2id[keypoint]]
        if not all([int(keypoint_x) >= face_bbox[0],
                    int(keypoint_y) >= face_bbox[1],
                    int(keypoint_x) <= face_bbox[2],
                    int(keypoint_y) <= face_bbox[3]]):
            does_match = False
            break
    return does_match


if __name__ == '__main__':
    pose_config = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    pose_checkpoint = '/home/bar_cohen/D-KinderGuardian/checkpoints/mmpose-hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    device = 'cuda:0'
    person_results = [{'bbox': [0, 0, 97, 269]}]
    image_name = '/home/bar_cohen/D-KinderGuardian/models/0001_c1_f0413118.jpg'

    # init pose estimation model using checkpoint
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device.lower())
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    img = Image.open(image_name).convert('RGB')

    # init face detection model
    face_detector = FaceDetector(keep_all=True, device=device)

    # get pose estimation
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model, np.array(img)[:, :, ::-1], person_results, bbox_thr=None,
                                                                   format='xywh',dataset_info=dataset_info)

    # detect faces
    face_bboxes, probs = face_detector.facenet_detecor.detect(img=img)
    face_bbox = []

    # visualize pose estimation and detected face
    if len(face_bboxes) > 0:
        face_bbox = [int(x) for x in face_bboxes[0]]
        img1 = ImageDraw.Draw(img)
        img1.rectangle(face_bbox, outline="red")
    out_file = os.path.join('/home/bar_cohen/D-KinderGuardian/models', f'vis_{image_name.split("/")[-1]}')
    vis_pose_result(pose_model, np.array(img)[:, :, ::-1], pose_results, dataset_info=dataset_info, out_file=out_file)

    # check if detected face matches pose estimation
    if face_bbox:
        if does_face_match_to_pose(pose_results, face_bbox):
            print('Detected face matches the main person in the crop!')
        else:
            print('Face is not of the main person in the crop!')

    print('daniel')

