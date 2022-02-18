import csv
import os
import pickle
import tempfile
from argparse import ArgumentParser, REMAINDER
import sys
from collections import defaultdict
from matplotlib import pyplot as plt
import tqdm
import torch
import numpy as np
import mmcv
import torch.nn.functional as F

# from DataProcessing.dataHandler import Crop
from DataProcessing.DB.dal import *
from DataProcessing.dataProcessingConstants import ID_TO_NAME
from FaceDetection.faceClassifer import FaceClassifer
from FaceDetection.faceDetector import FaceDetector
from DataProcessing.utils import viz_DB_data_on_video

sys.path.append('fast-reid')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader
from demo.predictor import FeatureExtractionDemo
from mmtrack.apis import inference_mot, init_model

CAM_ID = 1


def get_args():
    parser = ArgumentParser()
    parser.add_argument('track_config', help='config file for the tracking model')
    parser.add_argument('reid_config', help='config file for the reID model')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument('--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--track_checkpoint', help='checkpoint file for the track model')
    parser.add_argument('--reid_checkpoint', help='checkpoint file for the reID model')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--show', action='store_true', help='whether show the results on the fly')
    parser.add_argument('--backend', choices=['cv2', 'plt'], default='cv2', help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument('--crops_folder', help='Path to the folder in which the generated crops should be saved')
    parser.add_argument("--reid_opts", help="Modify reid-config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=REMAINDER, )
    parser.add_argument("--acc_th", help="The accuracy threshold that should be used for the tracking model",
                        default=0.8)
    parser.add_argument('--inference_only', action='store_true', help='use the tracking and reid model for inference')
    args = parser.parse_args()
    return args


def set_reid_cfgs(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.reid_config)
    cfg.merge_from_list(args.reid_opts)
    cfg.freeze()
    return cfg


def apply_reid_model(reid_model, data):
    feats = []
    pids = []
    camids = []
    print('Converting test data to feature vectors:')
    # Converts all images in the bounding_box_test and query to feature vectors
    for (feat, pid, camid) in tqdm.tqdm(reid_model.run_on_loader(data), total=len(data)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    print("The size of test gallery is", len(pids))
    feats = torch.cat(feats, dim=0)
    g_feat = feats
    g_pids = np.asarray(pids)
    g_camids = np.asarray(camids)
    return feats, g_feat, g_pids, g_camids


def find_best_reid_match(q_feat, g_feat, g_pids):
    """
    Given feature vectors of the query images, return the ids of the images that are most similar in the test gallery
    """
    features = F.normalize(q_feat, p=2, dim=1)
    others = F.normalize(g_feat, p=2, dim=1)
    distmat = 1 - torch.mm(features, others.t())
    distmat = distmat.numpy()
    best_match_in_gallery = np.argmin(distmat, axis=1)
    return g_pids[best_match_in_gallery] , distmat


def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    acc = result['track_results'][0][:, -1]
    mask = np.where(acc > acc_threshold)
    result['track_results'][0] = result['track_results'][0][mask]
    result['bbox_results'][0] = result['bbox_results'][0][mask]
    return result


def reid_inference(reid_model, img, result, frame_id, crops_folder=None):
    crops_bboxes = result['track_results'][0][:, 1:-1]
    crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
    q_feat = torch.empty((len(crops_imgs), 2048))
    for j in range(len(crops_imgs)):
        crop = np.array(crops_imgs[j])
        q_feat[j] = reid_model.run_on_image(crop)
        if crops_folder:
            os.makedirs(crops_folder, exist_ok=True)
            mmcv.imwrite(crop, os.path.join(crops_folder, f'frame_{frame_id}_crop_{j}.jpg'))
    return q_feat


def reid_track_inference(reid_model, track_imgs: list):
    q_feat = torch.empty((len(track_imgs), 2048))
    for j, crop in enumerate(track_imgs):
        crop = np.array(crop)
        q_feat[j] = reid_model.run_on_image(crop)
    return q_feat


def replace_ids(result, q_feat, g_feat, g_pids):
    """
    Replace the ids given by the tracking model with the ids computed by the re-id model
    """
    reid_ids = find_best_reid_match(q_feat, g_feat, g_pids)
    for k in range(len(result['track_results'][0])):
        result['track_results'][0][k][0] = reid_ids[k]


def crop_top_third(crop_img):
    y,x = crop_img.shape
    crop_img = crop_img[y*0.7:y, x*0.7:x]
    mmcv.imshow(crop_img)
    return crop_img

def create_data_by_re_id_and_track():
    """
    This function takes a video and runs both tracking, face-id and re-id models to create and label tracklets
    within the video. Note that for the video_name we skip the first 8 chars as the fit the IP_Camera video name
    convention, if entering a different video name note this and adpat your name accordingly.
    Returns: None. Creates and saves Crops according to --crops_folder arg

    """
    args = get_args()
    print(f'Args: {args}')
    db_location = DB_LOCATION
    if args.inference_only:
        print('*** Running in inference-only mode ***')
        db_location = '/mnt/raid1/home/bar_cohen/inference_db5.db'
        create_table(db_location)
        print(f'Created temp DB in: {db_location}')
    else:
        print(f'Saving the output crops to: {args.crops_folder}')
        assert args.crops_folder, "You must insert crop_folder param in order to create data"

    faceDetector = FaceDetector()
    le = pickle.load(open('/home/bar_cohen/KinderGuardian/FaceDetection/data/le.pkl', 'rb'))
    faceClassifer = FaceClassifer(num_classes=19, label_encoder=le)

    faceClassifer.model_ft.load_state_dict(
        torch.load("/home/bar_cohen/KinderGuardian/FaceDetection/checkpoints/best_model3.pth"))
    faceClassifer.model_ft.eval()

    reid_cfg = set_reid_cfgs(args)

    # build re-id test set. NOTE: query dir of the dataset should be empty!
    test_loader, num_query = build_reid_test_loader(reid_cfg,
                                                    dataset_name='DukeMTMC')  # will take the dataset given as argument

    # build re-id inference model:
    reid_model = FeatureExtractionDemo(reid_cfg, parallel=True)

    # run re-id model on all images in the test gallery and query folders:
    feats, g_feats, g_pids, g_camids = apply_reid_model(reid_model, test_loader)

    # initialize tracking model:
    tracking_model = init_model(args.track_config, args.track_checkpoint, device=args.device)

    # load images:
    imgs = mmcv.VideoReader(args.input)

    # iterate over all images and collect tracklets
    print('******* Creating tracklets: *******')

    # initialize a dictionary that will hold all the crops according to tracks
    # key: track_id, value: a dict representing a crop in the track, consists of: Crop object, crop img, face img
    tracklets = defaultdict(list)

    for image_index, img in tqdm.tqdm(enumerate(imgs), total=len(imgs)):
        if isinstance(img, str):
            img = os.path.join(args.input, img)
        result = tracking_inference(tracking_model, img, image_index, acc_threshold=float(args.acc_th))
        ids = list(map(int, result['track_results'][0][:, 0]))
        confs = result['track_results'][0][:, -1]
        crops_bboxes = result['track_results'][0][:, 1:-1]
        crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
        for i, (id, conf, crop_im) in enumerate(zip(ids, confs, crops_imgs)):
            face_img = faceDetector.facenet_detecor(crop_im)
            # for video_name we skip the first 8 chars as to fit the IP_Camera video name convention, if entering
            # a different video name note this.
            x1, y1, x2, y2 = list(map(int, crops_bboxes[i]))  # convert the bbox floats to ints
            crop = Crop(vid_name=args.input.split('/')[-1][8:-4],
                        frame_num=image_index,
                        track_id=id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        conf=conf,
                        cam_id=CAM_ID,
                        crop_id=-1,
                        reviewed_one=False,
                        reviewed_two=False,
                        invalid=False,
                        is_vague=False)
            crop.set_im_name()
            tracklets[id].append({'crop_img': crop_im, 'face_img': face_img, 'Crop': crop})

    print('******* Making predictions and saving crops to DB *******')
    db_entries = []

    if args.inference_only:
        correct = 0
        total_crops = 0
    else:
        os.makedirs(args.crops_folder, exist_ok=True)

    # iterate over all tracklets and make a prediction for every tracklet
    for track_id, crop_dicts in tqdm.tqdm(tracklets.items(), total=len(tracklets.keys())):
        track_imgs = [crop_dict.get('crop_img') for crop_dict in crop_dicts]
        q_feats = reid_track_inference(reid_model=reid_model, track_imgs=track_imgs)
        reid_ids, distmat = find_best_reid_match(q_feats, g_feats, g_pids)
        # reid_ids = np.where(reid_ids[np.min(distmat,axis=1)] <0.02) # todo min dist task
        bincount = np.bincount(reid_ids)
        reid_maj_vote = np.argmax(bincount)
        reid_maj_conf = bincount[reid_maj_vote] / len(reid_ids)
        label = ID_TO_NAME[reid_maj_vote]

        face_imgs = [crop_dict.get('face_img') for crop_dict in crop_dicts if crop_dict.get('face_img')
                     is not None and crop_dict.get('face_img') is not crop_dict.get('face_img').numel()]
        if len(face_imgs) > 0:  # at least 1 face was detected
            face_classifer_preds = faceClassifer.predict(torch.stack(face_imgs))
            bincount_face = torch.bincount(face_classifer_preds.cpu())
            face_label = ID_TO_NAME[faceClassifer.le.inverse_transform([int(torch.argmax(bincount_face))])[0]]
            if len(face_imgs) > 1:
                # if face_label == 'Adam':
                #     faceClassifer.imshow(face_imgs[0:2], labels=[face_label] * 2)
                pass  # uncomment above to show faces
            print(face_label)
            print(f'reid label: {label}, face label: {face_label}')
            print(f'do the predictors agree? f{label == face_label}')

            if reid_maj_conf <= 0.5: # silly heuristic todo do this according to the prob of the faceid model
                # print(f'do the predictors agree? f{label==face_label}')
                print("Take Faceanyways due to low conf rom reid")
                label = face_label

        # update missing info of the crop: crop_id, label and is_face, save the crop to the crops_folder and add to DB
        for crop_id, crop_dict in enumerate(crop_dicts):
            crop = crop_dict.get('Crop')
            crop.crop_id = crop_id
            crop.label = label

            if args.inference_only:
                tagged_label_crop = get_entries(filters={Crop.im_name == crop.im_name, Crop.invalid == False}).all()
                # print(f'DB label is: {tagged_label}, Inference label is: {reid_ids[crop_id]}')
                if tagged_label_crop: # there is a tagging for this crop which is not invalid, count it
                    total_crops += 1
                    tagged_label = tagged_label_crop[0].label
                    if tagged_label == label:
                        correct += 1

            face_img = crop_dict.get('face_img')
            crop.is_face = face_img is not None and face_img is not face_img.numel()
            if not args.inference_only:
                mmcv.imwrite(crop_dict['crop_img'], os.path.join(args.crops_folder, crop.im_name))
            db_entries.append(crop)

    add_entries(db_entries, db_location)

    if args.inference_only:
        acc = {correct / total_crops}
        print(f'Total accuracy: {acc}')
        print('Making visualization using temp DB')
        viz_DB_data_on_video(input_vid=args.input, output_path=args.output, DB_path=db_location,eval=True)
        assert db_location != DB_LOCATION, 'Pay attention! you almost destroyed the labeled DB!'
        os.remove(db_location)
    print("Done")


if __name__ == '__main__':
    create_data_by_re_id_and_track()



