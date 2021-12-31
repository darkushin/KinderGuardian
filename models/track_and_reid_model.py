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

from DataProcessing.dataProcessingConstants import ID_TO_NAME
from FaceDetection.faceClassifer import FaceClassifer
from FaceDetection.faceDetector import FaceDetector

sys.path.append('fast-reid')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader
from demo.predictor import FeatureExtractionDemo
from mmtrack.apis import inference_mot, init_model

class Crop:
    def __init__(self, frame_id:int,
                 bbox:np.array,
                 crop_img:torch.tensor,
                 face_img:torch.tensor,
                 track_id:int,
                 cam_id:int,
                 crop_id:int,
                 video_name:str):
        self.frame_id = frame_id
        self.bbox = bbox
        self.crop_img = crop_img
        self.face_img = face_img
        self.track_id = track_id
        self.cam_id = cam_id
        self.crop_id = crop_id
        self.video_name = video_name
        self.label = None
        self.unique_crop_id = None
        self.unique_face_crop_id = None
        self.update_hash()

    def set_label(self, label):
        self.label = label
        self.update_hash() # update unique_hash

    def update_hash(self):
        self.unique_id = f'video_name:{self.video_name[9:]}_track_id:{self.track_id}_cam_id:{self.cam_id}_frame_id:{self.frame_id}_crop_id:{self.crop_id}_label:{self.label}'
        if self.check_if_face_img():
            self.unique_face_crop_id = f'video_name:{self.video_name[9:]}_track_id:{self.track_id}_cam_id:{self.cam_id}_frame_id:{self.frame_id}_crop_id:{self.crop_id}__face_version__label:{self.label}'

    def save_crop(self, datapath):
        mmcv.imwrite(self.crop_img, os.path.join(datapath, f'{self.unique_id}.png'))
        if self.unique_face_crop_id and self.check_if_face_img():
            mmcv.imwrite(self.face_img.numpy(), os.path.join(datapath, f'{self.unique_face_crop_id}.png'))


    def check_if_face_img(self):
        return self.face_img is not None and self.face_img is not self.face_img.numel()
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
    parser.add_argument("--reid_opts", help="Modify reid-config options using the command-line 'KEY VALUE' pairs", default=[], nargs=REMAINDER,)
    parser.add_argument("--acc_th", help="The accuracy threshold that should be used for the tracking model", default=0.8)
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
    return g_pids[best_match_in_gallery]


def tracking_inference(tracking_model, img, frame_id, acc_threshold=0.8):
    result = inference_mot(tracking_model, img, frame_id=frame_id)
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

def reid_track_inference(reid_model, track_imgs:list):
    q_feat = torch.empty((len(track_imgs), 2048))
    for j,crop in enumerate(track_imgs):
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

def create_data_by_re_id_and_track():
    args = get_args()
    print(args.crops_folder)
    print(args)
    assert args.crops_folder , "You must insert crop_folder param in order to create data"

    faceDetector = FaceDetector()
    faceClassifer = FaceClassifer(num_classes=19)
    faceClassifer.model_ft.load_state_dict(torch.load('/home/bar_cohen/KinderGuardian/FaceDetection/best_model3.pth'))


    reid_cfg = set_reid_cfgs(args)

    # build re-id test set. NOTE: query dir of the dataset should be empty!
    test_loader, num_query = build_reid_test_loader(reid_cfg, dataset_name='DukeMTMC')  # will take the dataset given as argument

    # build re-id inference model:
    reid_model = FeatureExtractionDemo(reid_cfg, parallel=True)

    # run re-id model on all images in the test gallery and query folders:
    feats, g_feat, g_pids, g_camids = apply_reid_model(reid_model, test_loader)

    # initialize tracking model:
    tracking_model = init_model(args.track_config, args.track_checkpoint, device=args.device)

    # load images and set temp folders for output creation:
    imgs = mmcv.VideoReader(args.input)
    fps = int(imgs.fps)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    _out = args.output.rsplit('/', 1)
    if len(_out) > 1:
        os.makedirs(_out[0], exist_ok=True)

    # iterate over all images and collect tracklets
    print('create tracklets')
    tracklets = defaultdict(list)
    for image_index, img in enumerate(imgs):
        if isinstance(img, str):
            img = os.path.join(args.input, img)
        result = tracking_inference(tracking_model, img, image_index, acc_threshold=float(args.acc_th))
        ids = result['track_results'][0][:,0]
        crops_bboxes = result['track_results'][0][:, 1:-1]
        crops_imgs = mmcv.image.imcrop(img, crops_bboxes, scale=1.0, pad_fill=None)
        for i, (id, crop) in enumerate(zip(ids,crops_imgs)):
            face_img = faceDetector.facenet_detecor(crop)
            if face_img is not None and face_img is not face_img.numel():
                face_img = face_img.permute(1, 2, 0).int()
            crop_obj = Crop(video_name=args.input ,
                            frame_id=id,
                            bbox=crops_bboxes[i],
                            crop_img=crop,
                            face_img=face_img,
                            track_id=id,
                            cam_id=1,
                            crop_id=i)
            tracklets[id].append(crop_obj)

    print('make prediction and save crop')
    crops_db = []
    os.makedirs(args.crops_folder, exist_ok=True)
    for track_id, crops in tracklets.items():
        track_imgs = [crop.crop_img for crop in crops]
        if len(track_imgs) < 5: #todo add this as a param
            continue
        q_feat = reid_track_inference(reid_model=reid_model, track_imgs=track_imgs)
        reid_ids = find_best_reid_match(q_feat, g_feat, g_pids)
        bincount = np.bincount(reid_ids)
        reid_maj_vote = np.argmax(bincount)
        reid_maj_conf = bincount[reid_maj_vote] / len(reid_ids)
        label = ID_TO_NAME[reid_maj_vote]

        # faceClassifer.imshow(track_imgs, labels=[ID_TO_NAME[reid_maj_vote]] * len(track_imgs))
        # plt.imshow(track_imgs[0])
        # plt.title(ID_TO_NAME[reid_maj_vote])
        # plt.show()
        face_imgs = [crop.face_img for crop in crops if crop.check_if_face_img()]
        if len(face_imgs) > 0: # at least 1 face was detected
            face_classifer_preds = faceClassifer.predict(torch.cat(face_imgs), out=torch.tensor(len(face_imgs), face_imgs[0].shape[0],face_imgs[0].shape[1]))
            bincount_face = np.bincount(face_classifer_preds)
            face_label = ID_TO_NAME[np.argmax(bincount_face)]
            print(face_label)
            plt.imshow(face_imgs[0])
            plt.show()

            if reid_maj_conf < 0.5: # silly heuristic
                label = face_label

        for crop in crops:
            crop.set_label(label)
            crop.save_crop(datapath=args.crops_folder)
            del crop.crop_img # we dont want to keep this info in the crop obj
            del crop.face_img
            crops_db.append(crop)
        # todo insert faceid


    pickle.dump(crops_db, open(f'{args.crops_folder}_crop_db.pkl','wb'))
    print("Done")

def main():
    args = get_args()
    reid_cfg = set_reid_cfgs(args)

    # build re-id test set. NOTE: query dir of the dataset should be empty!
    test_loader, num_query = build_reid_test_loader(reid_cfg, dataset_name='DukeMTMC')  # will take the dataset given as argument

    # build re-id inference model:
    reid_model = FeatureExtractionDemo(reid_cfg, parallel=True)

    # run re-id model on all images in the test gallery and query folders:
    feats, g_feat, g_pids, g_camids = apply_reid_model(reid_model, test_loader)

    # initialize tracking model:
    tracking_model = init_model(args.track_config, args.track_checkpoint, device=args.device)

    # load images and set temp folders for output creation:
    imgs = mmcv.VideoReader(args.input)
    fps = int(imgs.fps)
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name
    _out = args.output.rsplit('/', 1)
    if len(_out) > 1:
        os.makedirs(_out[0], exist_ok=True)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # iterate over all images and run tracking and reid for every image:
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = os.path.join(args.input, img)

        result = tracking_inference(tracking_model, img, i, acc_threshold=float(args.acc_th))

        q_feat = reid_inference(reid_model, img, result, frame_id=i, crops_folder=args.crops_folder)

        # replace tracking ids with re-id ids
        replace_ids(result, q_feat, g_feat, g_pids)

        prog_bar.update()

        # save the image to the temp folder
        out_file = os.path.join(temp_path, f'{i:06d}.jpg')
        tracking_model.show_result(
            img,
            result,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)

    print(f'making the output video at {args.output} with a FPS of {fps}')
    mmcv.frames2video(temp_path, args.output, fps=fps, fourcc='mp4v')
    temp_dir.cleanup()


if __name__ == '__main__':
    create_data_by_re_id_and_track()

