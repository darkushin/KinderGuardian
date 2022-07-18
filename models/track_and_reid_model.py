import csv
import os
import pickle
import tempfile
import time
from argparse import ArgumentParser, REMAINDER
import sys
from collections import defaultdict, Counter
from torchvision.transforms import transforms
from PIL import Image
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import tqdm
import torch
import numpy as np
import mmcv
import torch.nn.functional as F
from DataProcessing.DB.dal import *
from DataProcessing.dataProcessingConstants import ID_TO_NAME, NAME_TO_ID
from FaceDetection.arcface import ArcFace
from FaceDetection.augmentions import normalize_image
# from FaceDetection.faceClassifer import FaceClassifer # uncomment to use FaceNet classifier
from FaceDetection.faceDetector import FaceDetector, is_img
from FaceDetection.pose_estimator import PoseEstimator
from DataProcessing.utils import viz_DB_data_on_video
from models.model_constants import ID_NOT_IN_VIDEO
import warnings
warnings.filterwarnings('ignore')
sys.path.append('fast-reid')
sys.path.append('centroids_reid')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader
from demo.predictor import FeatureExtractionDemo
from mmtrack.apis import inference_mot, init_model
from double_id_handler import remove_double_ids, NODES_ORDER
from CTL_reid_inference import *

CAM_ID = 1
P_POWER = 2
FAST_PICKLES = '/home/bar_cohen/raid/OUR_DATASETS/FAST_reid'
ABLATION_OUTPUT = '/mnt/raid1/home/bar_cohen/42Street/Results/42Street_inference_results.csv'
ABLATION_COLUMNS = ['description', 'video_name', 'ids_in_video', 'total_ids_in_video', 'total_tracks',
                    'tracks_with_face', 'pure_reid_model', 'reid_with_maj_vote', 'face_clf_only',
                    'face_clf_only_tracks_with_face', 'reid_with_face_clf_maj_vote', 'rank-1', 'sorted-rank-1',
                    'appearance-order', 'max-difference', 'model_name', 'running_time', 'total_crops']#,
                    # 'Adam', 'Avigail', 'Ayelet', 'Bar', 'Batel', 'Big-Gali', 'Eitan', 'Gali', 'Guy', 'Halel', 'Lea',
                    # 'Noga', 'Ofir', 'Omer', 'Roni', 'Sofi', 'Sofi-Daughter', 'Yahel', 'Hagai', 'Ella', 'Daniel']
ABLATION_COLUMNS.extend(list(NAME_TO_ID.keys()))

def get_args():
    parser = ArgumentParser()
    parser.add_argument('track_config', help='config file for the tracking model')
    parser.add_argument('reid_config', help='config file for the reID model')
    parser.add_argument('--pose_config', help='config file for the pose estimation model')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument('--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--track_checkpoint', help='checkpoint file for the track model')
    parser.add_argument('--reid_checkpoint', help='checkpoint file for the reID model')
    parser.add_argument('--pose_checkpoint', help='checkpoint file for the pose estimation model')
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
    parser.add_argument('--db_tracklets', action='store_true', help='use the tagged DB to create tracklets for inference')
    parser.add_argument('--experiment_mode', action='store_true', help='run in experiment_mode')
    parser.add_argument('--exp_description', help='The description of the experiment that should appear in the ablation study output')
    parser.add_argument('--reid_model', choices=['fastreid', 'ctl'], default='fastreid', help='Reid model that should be used.')
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


def get_reid_score(track_im_conf, distmat, g_pids):
    min_dist_squared = (np.min(distmat, axis=1) ** P_POWER + 1)
    best_match_scores = track_im_conf / min_dist_squared
    best_match_in_gallery = np.argmin(distmat, axis=1)
    ids_score = {pid : 0 for pid in ID_TO_NAME.keys()}
    for pid,score in zip(g_pids[best_match_in_gallery], best_match_scores): # an id chosen as the best match
        ids_score[pid] += score / track_im_conf.shape[0]

    return ids_score

def get_reid_score_all_ids_full_gallery(track_im_conf, simmat, g_pids):
    ids_score = {pid: 0 for pid in ID_TO_NAME.keys()}
    aligned_simmat = (track_im_conf[:, np.newaxis] * simmat) ** P_POWER
    aligned_simmat = simmat ** P_POWER
    for pid in set(g_pids):
        ids_score[pid] += aligned_simmat[:, g_pids[g_pids==pid]].max()
    return ids_score

def get_reid_score_mult_ids(track_im_conf, simmat, g_pids):
    print(simmat.shape)
    ids_score = {pid: 0 for pid in ID_TO_NAME.keys()}
    aligned_simmat = (track_im_conf[:, np.newaxis] * simmat) ** P_POWER
    scores = aligned_simmat.sum(axis=0) / len(track_im_conf)
    for pid, score in  zip(g_pids, scores):
        ids_score[pid] += score
    return ids_score


def find_best_reid_match(q_feat, g_feat, g_pids, track_imgs_conf):
    """
    Given feature vectors of the query images, return the ids of the images that are most similar in the test gallery
    """
    features = F.normalize(q_feat, p=2, dim=1)
    others = F.normalize(g_feat, p=2, dim=1)
    simmat = torch.mm(features, others.t()).numpy()
    distmat = 1 - simmat
    # distmat = distmat.numpy()
    # ids_score = get_reid_score(track_imgs_conf, distmat, g_pids)
    ids_score = get_reid_score_all_ids_full_gallery(track_imgs_conf, simmat, g_pids)
    # ids_score = get_reid_score_mult_ids(track_imgs_conf, simmat, g_pids)
    # ids_score = get_reid_score_cosine_sim(track_imgs_conf, simmat, g_pids)
    best_match_in_gallery = np.argmin(distmat, axis=1)
    return g_pids[best_match_in_gallery] , ids_score


def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.98):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
    result['track_results'] = result['track_bboxes']
    result['bbox_results'] = result['det_bboxes']
    return result


def reid_track_inference(reid_model, track_imgs: list):
    q_feat = torch.empty((len(track_imgs), 2048))
    for j, crop in enumerate(track_imgs):
        crop = np.array(crop)
        q_feat[j] = reid_model.run_on_image(crop)
    return q_feat


def get_face_score(faceClassifer, preds,probs, detector_conf):
    face_id_scores = {pid : 0 for pid in ID_TO_NAME.keys()}
    for pid, prob, conf in zip(preds, probs, detector_conf):
        real_pid = int(faceClassifer.le.inverse_transform([int(pid)])[0])
        face_id_scores[real_pid] += ((float(prob[pid]) * conf)**P_POWER) / len(preds)
    return face_id_scores

def get_face_score_all_ids(faceClassifer,track_probs, detector_conf):
    face_id_scores = {pid: 0 for pid in ID_TO_NAME.keys()}
    for crop_probs, conf in zip(track_probs, detector_conf):
        for pid, prob_i in enumerate(crop_probs):
            real_pid = int(faceClassifer.le.inverse_transform([int(pid)])[0])
            face_id_scores[real_pid] += ((float(prob_i) * conf) ** P_POWER) / len(detector_conf)
    return face_id_scores

def gen_reid_features(reid_cfg, reid_model, output_path=None):
    test_loader, num_query = build_reid_test_loader(reid_cfg,
                                                    dataset_name='DukeMTMC')  # will take the dataset given as argument
    feats, g_feats, g_pids, g_camids = apply_reid_model(reid_model, test_loader)
    if not output_path:
        output_path = "/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/"
    print('dumping gallery to pickles....:')
    with open(os.path.join(output_path, 'feats'), 'wb') as feats_filer:
        pickle.dump(feats, feats_filer)
    with open(os.path.join(output_path, 'g_feats'), 'wb') as g_feats_filer:
        pickle.dump(g_feats,g_feats_filer)
    with open(os.path.join(output_path, 'g_pids'), 'wb') as g_pids_filer:
        pickle.dump(g_pids, g_pids_filer)
    with open(os.path.join(output_path, 'g_camids'), 'wb') as g_cam_filer:
        pickle.dump(g_camids, g_cam_filer)


def load_reid_features(output_path=None):
    print('loading gallery from pickles....:')
    if not output_path:
        output_path = "/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/"
    with open(os.path.join(output_path, 'feats'), 'rb') as feats_filer:
      feats = pickle.load(feats_filer)
    with open(os.path.join(output_path, 'g_feats'), 'rb') as g_feats_filer:
      g_feats = pickle.load(g_feats_filer)
    with open(os.path.join(output_path, 'g_pids'), 'rb') as g_pids_filer:
      g_pids = pickle.load(g_pids_filer)
    with open(os.path.join(output_path, 'g_camids'), 'rb') as g_cam_filer:
      g_camids = pickle.load(g_cam_filer)
    return feats, g_feats, g_pids, g_camids


def write_ablation_results(args, columns_dict, total_crops, total_crops_of_tracks_with_face, ids_acc_dict, ablation_df, db_location):
        acc_columns = ['pure_reid_model', 'face_clf_only', 'reid_with_maj_vote', 'reid_with_face_clf_maj_vote']
        acc_columns.extend(NODES_ORDER)
        for acc in acc_columns:
            columns_dict[acc] = columns_dict[acc] / total_crops

        if total_crops_of_tracks_with_face > 0:
            columns_dict['face_clf_only_tracks_with_face'] = columns_dict['face_clf_only_tracks_with_face'] / total_crops_of_tracks_with_face
        ids_set = set(name for name, value in ids_acc_dict.items() if value[0] > 0)
        columns_dict['total_ids_in_video'] = len(ids_set)
        columns_dict['ids_in_video'] = str(ids_set)
        columns_dict['description'] = args.exp_description if args.exp_description else ""
        columns_dict['total_crops'] = total_crops
        for name, value in ids_acc_dict.items():
            if value[0] == ID_NOT_IN_VIDEO: # this id was never in video
                columns_dict[name] = ID_NOT_IN_VIDEO
            elif value[0] > 0: # id was found in video but never correctly classified
                columns_dict[name] = value[1] / value[0]
        ablation_df.append(columns_dict, ignore_index=True).to_csv(ABLATION_OUTPUT)
        # print('Making visualization using temp DB')
        # viz_DB_data_on_video(input_vid=args.input, output_path=args.output, DB_path=db_location,eval=True)
        assert db_location != DB_LOCATION, 'Pay attention! you almost destroyed the labeled DB!'
        print('removing temp DB')
        os.remove(db_location)


def create_tracklets_from_db(vid_name, args):
    """
    Given a video name, create the tracklets for this video using the tagged DB.
    The returned tracklets are a dictionary where the key is the track_id, and the value is a list of dictionaries so
    that every dictionary contains the 'crop_img', 'face_img', 'Crop' and 'face_img_conf' of a single crop in the track.
    """
    tracklets = defaultdict(list)
    face_detector = FaceDetector(keep_all=True, device=args.device)
    if args.pose_config:  # todo: after using this by default remove this if and the warning
        pose_estimator = PoseEstimator(args.pose_config, args.pose_checkpoint, args.device)
    else:
        pose_estimator = None
        print('Warning! not using pose estimation when building tracks.')

    tracks = [track.track_id for track in get_entries(filters=({Crop.vid_name == vid_name}), group=Crop.track_id).all()]
    for track in tracks:
        crops = get_entries(filters=(Crop.vid_name == vid_name, Crop.track_id == track, Crop.invalid == False)).all()
        for temp_crop in crops:
            crop = Crop(vid_name=temp_crop.vid_name,
                        frame_num=temp_crop.frame_num,
                        track_id=temp_crop.track_id,
                        x1=temp_crop.x1, y1=temp_crop.y1, x2=temp_crop.x2, y2=temp_crop.y2,
                        conf=temp_crop.conf,
                        cam_id=temp_crop.cam_id,
                        crop_id=temp_crop.crop_id,
                        is_face=temp_crop.is_face,
                        label=temp_crop.label,
                        reviewed_one=temp_crop.reviewed_one,
                        reviewed_two=temp_crop.reviewed_two,
                        invalid=temp_crop.invalid,
                        is_vague=temp_crop.is_vague)
            crop.set_im_name()
            im_path = f'/home/bar_cohen/raid/{vid_name}/{crop.im_name}'
            if not os.path.isfile(im_path):
                im_location = 'v' + crop.im_name[2:]
                im_path =  f'/home/bar_cohen/raid/{vid_name}/{im_location}'
            crop_im = Image.open(im_path)
            # added for CTL inference
            ctl_img = crop_im
            ctl_img.convert("RGB")

            face_bboxes, face_probs = face_detector.facenet_detecor.detect(img=crop_im)
            face_imgs = face_detector.facenet_detecor.extract(crop_im, face_bboxes, save_path=None)
            face_img, face_prob = None, 0
            if face_imgs is not None and len(face_imgs) > 0:
                if pose_estimator:
                    face_img, face_prob = pose_estimator.find_matching_face(crop_im, face_bboxes, face_probs, face_imgs)
                    # uncomment this if using FaceNet classifier as you need FaceNet
                    # if is_img(face_img):
                    #     face_img = normalize_image(face_img)
            crop_im = mmcv.imread(im_path)
            tracklets[track].append({'crop_img': crop_im, 'face_img': face_img, 'Crop': crop, 'face_img_conf': face_prob,
                                     'ctl_img': ctl_img})
    del face_detector
    return tracklets


def get_vid_name(args):
    # for Kinderguardian use the following uncommented:
    # KG_vid_name = args.input.split('/')[-1][9:-4]
    vid_name =  args.input.split('/')[-1][:-4]
    return vid_name



def create_tracklets_using_tracking(args):
    # initialize tracking model:
    tracking_model = init_model(args.track_config, args.track_checkpoint, device=args.device)
    face_detector = FaceDetector(keep_all=True, device=args.device)
    if args.pose_config:  # todo: after using this by default remove this if and the warning
        pose_estimator = PoseEstimator(args.pose_config, args.pose_checkpoint, args.device)
    else:
        pose_estimator = None
        print('Warning! not using pose estimation when building tracks.')

    # load images:
    imgs = mmcv.VideoReader(args.input)

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

            # face_img, face_prob = face_detector.get_single_face(crop_im, is_PIL_input=False)
            # face_prob = face_prob if face_prob else 0
            # for video_name we skip the first 8 chars as to fit the IP_Camera video name convention, if entering
            # a different video name note this.
            face_bboxes, face_probs = face_detector.facenet_detecor.detect(img=crop_im)
            face_imgs = face_detector.facenet_detecor.extract(crop_im, face_bboxes, save_path=None)
            face_img, face_prob = None, 0
            if face_imgs is not None and len(face_imgs) > 0:
                if pose_estimator:
                    face_img, face_prob = pose_estimator.find_matching_face(crop_im, face_bboxes, face_probs, face_imgs)
                    # if is_img(face_img): # return this if you want to go back to kinderguardian
                        # face_img = normalize_image(face_img)

            x1, y1, x2, y2 = list(map(int, crops_bboxes[i]))  # convert the bbox floats to intsP
            crop = Crop(vid_name=get_vid_name(args),
                        frame_num=image_index,
                        track_id=id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        conf=conf,
                        cam_id=CAM_ID,
                        crop_id=-1,
                        is_face=is_img(face_img),
                        reviewed_one=False,
                        reviewed_two=False,
                        invalid=False,
                        is_vague=False)
            crop.set_im_name()
            mmcv.imwrite(crop_im, '/mnt/raid1/home/bar_cohen/CTL_Reid/t.jpg')
            ctl_img = Image.open('/mnt/raid1/home/bar_cohen/CTL_Reid/t.jpg').convert("RGB")
            tracklets[id].append({'crop_img': crop_im, 'face_img': face_img, 'Crop': crop, 'face_img_conf': face_prob,
                                  'ctl_img': ctl_img})
    del face_detector
    return tracklets


def create_or_load_tracklets(args, create_tracklets:bool):
    if create_tracklets and not args.db_tracklets:  # create tracklets from video using tracking
        print('******* Creating tracklets using tracking: *******')
        tracklets = create_tracklets_using_tracking(args=args)
        pickle.dump(tracklets, open('/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/tracklets.pkl', 'wb'))

    elif create_tracklets and args.db_tracklets:  # create tracklets using the tagged DB
        print('***** Creating tracklets using DB *****')
        tracklets = create_tracklets_from_db(vid_name=get_vid_name(args), args=args)
        pickle.dump(tracklets, open('/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/db-tracklets.pkl', 'wb'))

    else:
        if args.db_tracklets:
            print('***** Using loaded DB tracklets! *****')
            tracklets = pickle.load(open('/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/db-tracklets.pkl', 'rb'))
        else:
            print('***** Using loaded tracking tracklets! *****')
            tracklets = pickle.load(open('/mnt/raid1/home/bar_cohen/OUR_DATASETS/pickles/tracklets.pkl', 'rb'))
    return tracklets


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
    columns_dict = {}
    start = time.time()
    if args.inference_only:
        if os.path.isfile(ABLATION_OUTPUT):
            ablation_df = pd.read_csv(ABLATION_OUTPUT, index_col=[0])
        else:
            ablation_df = pd.DataFrame(columns=ABLATION_COLUMNS)
        columns_dict = {k: 0 for k in ablation_df.columns}
        columns_dict['video_name'] = args.input.split('/')[-1]
        columns_dict['model_name'] = args.reid_model
        print('*** Running in inference-only mode ***')
        db_location = f'/mnt/raid1/home/bar_cohen/OUR_DATASETS/temp_db/inference_db{np.random.randint(0, 10000000000)}.db'
        if os.path.isfile(db_location): # remove temp db if leave-over from prev runs
            assert db_location != DB_LOCATION, 'Pay attention! you almost destroyed the labeled DB!'
            os.remove(db_location)
        create_table(db_location)
        print(f'Created temp DB in: {db_location}')
    else:
        print(f'Saving the output crops to: {os.path.join(args.crops_folder, get_vid_name(args))}')
        assert args.crops_folder, "You must insert crop_folder param in order to create data"

    with open("/mnt/raid1/home/bar_cohen/FaceData/le_19.pkl", 'rb') as f:
        le = pickle.load(f)

    if args.reid_model == 'fastreid':
        reid_cfg = set_reid_cfgs(args)

        # build re-id inference model:
        reid_model = FeatureExtractionDemo(reid_cfg, parallel=True)

        # run re-id model on all images in the test gallery and query folders:
        # build re-id test set. NOTE: query dir of the dataset should be empty!
        if not os.path.isdir(FAST_PICKLES):
            os.makedirs(FAST_PICKLES, exist_ok=True)
            gen_reid_features(reid_cfg, reid_model, output_path=FAST_PICKLES)  # UNCOMMENT TO Recreate reid features
        feats, g_feats, g_pids, g_camids = load_reid_features(output_path=FAST_PICKLES)
    else:
        # args.reid_config = "./centroids_reid/configs/256_resnet50.yml"
        reid_cfg = set_CTL_reid_cfgs(args)
        # initialize reid model:
        checkpoint = torch.load(reid_cfg.TEST.WEIGHT)
        checkpoint['hyper_parameters']['MODEL']['PRETRAIN_PATH'] = './centroids_reid/models/resnet50-19c8e357.pth'
        reid_model = CTLModel._load_model_state(checkpoint)

        # create gallery feature:
        # if not os.path.isdir(CTL_PICKLES):
        os.makedirs(CTL_PICKLES, exist_ok=True)
        gallery_data = make_inference_data_loader(reid_cfg, reid_cfg.DATASETS.ROOT_DIR, ImageDataset)
        g_feats, g_paths = create_gallery_features(reid_model, gallery_data, args.device, output_path=CTL_PICKLES)

        # OR load gallery feature:
        # g_feats, g_paths = load_gallery_features(gallery_path=CTL_PICKLES)
        g_pids = g_paths.astype(int)
        g_feats = torch.from_numpy(g_feats)

    tl_start = time.time()
    print('Starting to create tracklets')
    if args.experiment_mode:
        tracklets = create_or_load_tracklets(args,create_tracklets=True)
    else:
        if not args.db_tracklets:  # create tracklets from video using tracking
            tracklets = create_tracklets_using_tracking(args=args)
        else:  # create tracklets from DB
            tracklets = create_tracklets_from_db(vid_name=get_vid_name(args), args=args)
    tl_end = time.time()
    print(f'Total time for loading tracklets for video {args.input.split("/")[-1]}: {int(tl_end-tl_start)}')
    print('******* Making predictions and saving crops to DB *******')
    db_entries = []
    # id dict value at index 0 - number of times appeared in video
    # id dict value at index 1 - number of times correctly classified in video
    ids_acc_dict = {name: [ID_NOT_IN_VIDEO,ID_NOT_IN_VIDEO] for name in ID_TO_NAME.values()}

    total_crops = 0
    total_crops_of_tracks_with_face = 0
    if not args.inference_only:
        os.makedirs(os.path.join(args.crops_folder,get_vid_name(args)), exist_ok=True)

    all_tracks_final_scores = dict()

    # UnComment to use FaceNet FaceClassifier
    # faceClassifer = FaceClassifer(num_classes=19, label_encoder=le, device='cuda:0')
    # faceClassifer.model_ft.load_state_dict(torch.load("/mnt/raid1/home/bar_cohen/FaceData/checkpoints/4.8 Val, 1.pth"))
    # faceClassifer.model_ft.eval()

    arc = ArcFace(gallery_path="/mnt/raid1/home/bar_cohen/42street/clusters/")
    arc.read_gallery()



    # iterate over all tracklets and make a prediction for every tracklet
    for track_id, crop_dicts in tqdm.tqdm(tracklets.items(), total=len(tracklets.keys())):
        if args.inference_only:
            columns_dict['total_tracks'] += 1
        track_imgs_conf = np.array([crop_dict.get('Crop').conf for crop_dict in crop_dicts])
        if args.reid_model == 'fastreid':
            track_imgs = [crop_dict.get('crop_img') for crop_dict in crop_dicts]
            q_feats = reid_track_inference(reid_model=reid_model, track_imgs=track_imgs)
            reid_ids, reid_scores = find_best_reid_match(q_feats, g_feats, g_pids, track_imgs_conf)
        else:
            track_imgs = [crop_dict.get('ctl_img') for crop_dict in crop_dicts]
            # use loaded images for inference:
            q_feats = ctl_track_inference(model=reid_model, cfg=reid_cfg, track_imgs=track_imgs, device=args.device)
            q_feats = torch.from_numpy(q_feats)
            reid_ids, reid_scores = find_best_reid_match(q_feats, g_feats, g_pids, track_imgs_conf)

        # print(reid_scores)
        bincount = np.bincount(reid_ids)
        reid_maj_vote = np.argmax(bincount)
        reid_maj_conf = bincount[reid_maj_vote] / len(reid_ids)
        maj_vote_label = ID_TO_NAME[reid_maj_vote]

        final_label_id = max(reid_scores, key=reid_scores.get)
        final_label_conf = reid_scores[final_label_id]  # only reid at this point
        final_label = ID_TO_NAME[final_label_id]
        # print(final_label)
        all_tracks_final_scores[track_id] = reid_scores  # add reid scores in case the track doesn't include face images

        face_imgs = [crop_dict.get('face_img') for crop_dict in crop_dicts if is_img(crop_dict.get('face_img'))]
        face_imgs_conf = np.array([crop_dict.get('face_img_conf') for crop_dict in crop_dicts if crop_dict.get('face_img_conf') > 0])
        assert len(face_imgs) == len(face_imgs_conf)
        is_face_in_track = False
        face_label = 'None'
        if len(face_imgs) > 0:  # at least 1 face was detected
            is_face_in_track = True
            if args.inference_only:
                columns_dict['tracks_with_face'] += 1

            # FACENET --- uncomment this to use FaceNet classifier. Important Note, make sure to normilaze face images when using FaceNet
            # face_clf_preds, face_clf_outputs = faceClassifer.predict(torch.stack(face_imgs))
            # bincount_face = torch.bincount(face_clf_preds.cpu())
            # face_label = ID_TO_NAME[faceClassifer.le.inverse_transform([int(torch.argmax(bincount_face))])[0]]
            # face_scores = get_face_score_all_ids(faceClassifer, face_clf_outputs, face_imgs_conf)

            # ARCFACE --- the below model uses ArcFace as the Face Classifier
            face_scores = arc.predict_track(face_imgs)
            # print(face_scores)
            alpha = 0.49 # TODO enter as an arg
            final_scores = {pid : alpha*reid_score + (1-alpha) * face_score for pid, reid_score, face_score in zip(reid_scores.keys() , reid_scores.values(), face_scores.values())}
            print(final_scores)
            all_tracks_final_scores[track_id] = final_scores
            final_label = ID_TO_NAME[max(final_scores, key=final_scores.get)]
            print(final_label)
            # print(final_label)
        # update missing info of the crop: crop_id, label and is_face, save the crop to the crops_folder and add to DB

        for crop_id, crop_dict in enumerate(crop_dicts):
            crop = crop_dict.get('Crop')
            crop.crop_id = crop_id
            crop_label = ID_TO_NAME[reid_ids[crop_id]]
            crop.label = final_label
            if crop.conf >= float(args.acc_th):
                db_entries.append(crop)

            if args.inference_only and crop.conf >= float(args.acc_th):
                total_crops, total_crops_of_tracks_with_face = update_ablation_results(columns_dict, crop, crop_label,
                                                                                       face_label, final_label,
                                                                                       ids_acc_dict, is_face_in_track,
                                                                                       maj_vote_label, total_crops,
                                                                                       total_crops_of_tracks_with_face)

            if not args.inference_only:
                mmcv.imwrite(crop_dict['crop_img'], os.path.join(args.crops_folder,get_vid_name(args), crop.im_name))

    add_entries(db_entries, db_location)

    # handle double-id and update it in the DB
    # Remove double ids according to different heuristics and record it in the ablation study results
    # for nodes_order in NODES_ORDER:  # NOTE: the last order in this list will be used for the visualization
    #     new_id_dict = remove_double_ids(vid_name=args.input.split('/')[-1][9:-4], tracks_scores=all_tracks_final_scores,
    #                                     db_location=db_location, nodes_order=nodes_order)
    #     session = create_session(db_location)
    #     if args.inference_only:
    #         assert db_location != DB_LOCATION, 'You fool!'
    #     tracks = [track.track_id for track in get_entries(filters=(), group=Crop.track_id, db_path=db_location, session=session)]
    #     for track in tracks:
    #         crops = get_entries(filters=({Crop.track_id==track}), db_path=db_location, session=session).all()
    #         for crop in crops:
    #             crop.label = ID_2_CHAR[new_id_dict[track]]
    #             if args.inference_only:
    #                 tagged_label_crop = get_entries(filters={Crop.im_name == crop.im_name, Crop.invalid == False}).all()
    #                 if tagged_label_crop and tagged_label_crop[0].label == crop.label:
    #                     columns_dict[nodes_order] += 1

    # calculate new precision after IDs update and add to ablation study
    viz_DB_data_on_video(input_vid=args.input, output_path=args.output, DB_path=db_location,eval=False)

    # session.commit()
    end = time.time()
    # columns_dict['running_time'] = int(end-start)
    if args.inference_only and total_crops > 0:
        write_ablation_results(args, columns_dict, total_crops, total_crops_of_tracks_with_face, ids_acc_dict, ablation_df, db_location)

    print(f"Done within {int(end-start)} seconds.")


def update_ablation_results(columns_dict, crop, crop_label, face_label, final_label, ids_acc_dict, is_face_in_track,
                            maj_vote_label, total_crops, total_crops_of_tracks_with_face):
    tagged_label_crop = get_entries(filters={Crop.im_name == crop.im_name, Crop.invalid == False}).all()
    # print(f'DB label is: {tagged_label}, Inference label is: {reid_ids[crop_id]}')
    if tagged_label_crop:  # there is a tagging for this crop which is not invalid, count it
        total_crops += 1
        if is_face_in_track:
            total_crops_of_tracks_with_face += 1
        tagged_label = tagged_label_crop[0].label
        if ids_acc_dict[tagged_label][0] == ID_NOT_IN_VIDEO:  # init this id as present in vid
            ids_acc_dict[tagged_label][0] = 0
            ids_acc_dict[tagged_label][1] = 0
        ids_acc_dict[tagged_label][0] += 1

        if tagged_label == crop_label:
            columns_dict['pure_reid_model'] += 1
        if tagged_label == maj_vote_label:
            columns_dict['reid_with_maj_vote'] += 1
        if is_face_in_track and tagged_label == face_label:
            columns_dict['face_clf_only_tracks_with_face'] += 1
            columns_dict['face_clf_only'] += 1
        if tagged_label == final_label:
            columns_dict['reid_with_face_clf_maj_vote'] += 1
            ids_acc_dict[tagged_label][1] += 1
    return total_crops, total_crops_of_tracks_with_face


if __name__ == '__main__':
    create_data_by_re_id_and_track()



