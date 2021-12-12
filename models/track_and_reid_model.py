import os
import tempfile
from argparse import ArgumentParser, REMAINDER
import sys
import tqdm
import torch
import numpy as np
import mmcv
import torch.nn.functional as F

sys.path.append('fast-reid')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader
from demo.predictor import FeatureExtractionDemo
from mmtrack.apis import inference_mot, init_model



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


def replace_ids(result, q_feat, g_feat, g_pids):
    """
    Replace the ids given by the tracking model with the ids computed by the re-id model
    """
    reid_ids = find_best_match(q_feat, g_feat, g_pids)
    for k in range(len(result['track_results'][0])):
        result['track_results'][0][k][0] = reid_ids[k]


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
    main()

