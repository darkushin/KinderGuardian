import sys
sys.path.append('centroids_reid')
from centroids_reid.inference.inference_utils import ImageDataset, TrackDataset, make_inference_data_loader, calculate_centroids, create_pid_path_index
from centroids_reid.train_ctl_model import CTLModel
from centroids_reid.config import cfg
from centroids_reid.utils.reid_metric import get_dist_func
import torch
import numpy as np
import os
from model_runner import get_args
import pickle
import tqdm

# our files have the following path format: `/datasets/DukeMTMC-reID/query/0005_c2_f0046985.jpg`
extract_id = lambda x: x.rsplit("/", 1)[1].rsplit("_")[0]
CTL_PICKLES = '/home/bar_cohen/raid/CTL_Reid'


def _inference(model, batch, device, normalize_with_bn=True):
    """
    Inference method taken from ./centroids_reid/inference/inference_utils.py and modified to support device selection.
    The function receives the CTL model and a batch of images and creates feature vectors for the batch using the backbone.
    """
    model.eval()
    with torch.no_grad():
        data, _, filename = batch
        _, global_feat = model.backbone(data.cuda(device=device))
        if normalize_with_bn:
            global_feat = model.bn(global_feat)
        return global_feat, filename


def run_inference(model, val_loader, device=None):
    """
    Inference method taken from ./centroids_reid/inference/inference_utils.py and modified to support device selection.
    The function receives a data loader and creates embeddings for all the images in the data loader.
    """
    embeddings = []
    paths = []
    if device:
        device = int(device.split(':')[1])
        model = model.cuda(device=device)

    for x in val_loader:
        embedding, path = _inference(model, x, device)
        for vv, pp in zip(embedding, path):
            paths.append(pp)
            embeddings.append(vv.detach().cpu().numpy())

    embeddings = np.array(np.vstack(embeddings))
    paths = np.array(paths)
    return embeddings, paths


def set_CTL_reid_cfgs(args):
    cfg.merge_from_file(args.reid_config)
    cfg.merge_from_list(args.reid_opts)
    return cfg


def create_gallery_features(model, data_loader, device, output_path=None):
    """
    Create embeddings for the images given in the data_loader.
    This function is used to create embeddings for gallery images and optionally save them to a pickle.
    """
    print('Starting to create gallery feature vectors')
    g_feat, paths_gallery = run_inference(model, data_loader, device)
    if cfg.MODEL.USE_CENTROIDS:
        pid_path_index = create_pid_path_index(paths=paths_gallery, func=extract_id)
        g_feat, paths_gallery = calculate_centroids(g_feat, pid_path_index)
        print('Created gallery feature vectors using centroids.')
    else:
        paths_gallery = np.array([pid.split('/')[-1].split('_')[0] for pid in paths_gallery])  # need to be only the string id of a person ('0015' etc.)
        print('Did not use centroids for gallery feature vectors.')

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        print(f'Saving gallery feature vectors to: {output_path}')
        pickle.dump(g_feat, open(os.path.join(output_path, 'g_feats.pkl'), 'wb'))
        pickle.dump(paths_gallery, open(os.path.join(output_path, 'g_paths.pkl'), 'wb'))

    return g_feat, paths_gallery


def load_gallery_features(gallery_path):
    """
    Load previously create feature vectors of the gallery, from the given path.
    """
    if not gallery_path:
        raise Exception('Please provide a path to a folder from which the gallery embeddings should be loaded.')

    print('Loading gallery feature vectors from pickles...')
    g_feat = pickle.load(open(os.path.join(gallery_path, 'g_feats.pkl'), 'rb'))
    paths_gallery = pickle.load(open(os.path.join(gallery_path, 'g_paths.pkl'), 'rb'))

    return g_feat, paths_gallery


def ctl_track_inference(model, cfg, track_imgs, device):
    """
    Given loaded images of a track, run CTL inference on these images and return the embeddings of these images.
    """
    query_data = make_inference_data_loader(cfg, path='', dataset_class=TrackDataset, loaded_imgs=track_imgs)
    q_feats, _ = run_inference(model, query_data, device)
    return q_feats


#######################################################################
# Below are functions that are not part of the inference pipline.
# These functions can be used to verify the CTL model on our datasets.
#######################################################################

def create_distmat(q_feat, g_feat):
    """
    Compute the distance matrix of the query images from the gallery using cosine distance.
    """
    dist_func = get_dist_func('cosine')
    q_feat = torch.from_numpy(q_feat)
    g_feat = torch.from_numpy(g_feat)

    distmat = dist_func(x=q_feat, y=g_feat).cpu().numpy()
    return distmat


def create_query_prob_vector(q_feats, g_feats, g_paths):
    """
    Given the query and gallery feature vectors, compute the distmat between every query and the gallery and produce for
    every query image a probability vector.
    """
    distmat = create_distmat(q_feats, g_feats)

    # compute probabilities (ids with lower distance should receive higher probability) in the following way:
    # compute exp(-d) for every distance in the distmat and divide be the sum of each row
    distmat = np.exp(-distmat)
    probs_vectors = (distmat.transpose() / np.sum(distmat, axis=1)).transpose()
    assert np.sum(probs_vectors, axis=1) == 1, 'Probability vectors do not sum up to 1!'

    return probs_vectors


def compute_accuracy(q_feats, q_paths, g_feats, g_paths):
    distmat = create_distmat(q_feats, g_feats)
    indices = np.argsort(distmat, axis=1)
    
    accuracy = 0
    for i in range(len(q_feats)):
        min_idx = indices[i][0]
        true_label = extract_id(g_paths[min_idx])
        pred_label = extract_id(q_paths[i])

        if true_label == pred_label:
            accuracy += 1
    print(f'Total accuracy: {accuracy / len(q_feats)}')


def usage_example_folder_path_query(args):
    """
    Example of how to use the CTL model for inference given the args of the model_runner and the query folder of a reid
    dataset.
    """
    reid_cfg = set_CTL_reid_cfgs(args)

    # initialize reid model:
    checkpoint = torch.load(reid_cfg.TEST.WEIGHT)
    checkpoint['hyper_parameters']['MODEL']['PRETRAIN_PATH'] = './centroids_reid/models/resnet50-19c8e357.pth'
    reid_model = CTLModel._load_model_state(checkpoint)

    # create gallery feature:
    # gallery_imgs_path = '/home/bar_cohen/KinderGuardian/fast-reid/datasets/diff_day_train_as_test_0730_0808_quary/bounding_box_test'
    # gallery_data = make_inference_data_loader(reid_cfg, gallery_imgs_path, ImageDataset)
    # g_feats, g_paths = create_gallery_features(reid_model, gallery_data, args.device, output_path=CTL_PICKLES)

    # OR load gallery feature:
    g_feats, g_paths = load_gallery_features(gallery_path=CTL_PICKLES)
    
    # compute query feature vectors:
    query_imgs_path = '/home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0730/bounding_box_test'
    query_data = make_inference_data_loader(reid_cfg, query_imgs_path, ImageDataset)
    q_feats, q_paths = run_inference(reid_model, query_data, args.device)

    # compute accuracy for dataset:
    compute_accuracy(q_feats, q_paths, g_feats, g_paths)


def usage_example_track_as_query(args):
    """
    Example of how to use the CTL model for inference given the args of the model_runner and images of a track.
    """
    reid_cfg = set_CTL_reid_cfgs(args)

    # initialize reid model:
    checkpoint = torch.load(reid_cfg.TEST.WEIGHT)
    checkpoint['hyper_parameters']['MODEL']['PRETRAIN_PATH'] = './centroids_reid/models/resnet50-19c8e357.pth'
    reid_model = CTLModel._load_model_state(checkpoint)

    # create gallery feature:
    # gallery_imgs_path = reid_cfg.DATASETS.ROOT_DIR
    # gallery_data = make_inference_data_loader(reid_cfg, gallery_imgs_path, ImageDataset)
    # g_feats, g_paths = create_gallery_features(reid_model, gallery_data, args.device, output_path=CTL_PICKLES)

    # OR load gallery feature:
    g_feats, g_paths = load_gallery_features(gallery_path=CTL_PICKLES)

    # compute query feature vectors:
    track_imgs = pickle.load(open(os.path.join(CTL_PICKLES, 'track_imgs.pkl'), 'rb'))
    query_data = make_inference_data_loader(reid_cfg, path='', dataset_class=TrackDataset, loaded_imgs=track_imgs)
    q_feats, q_paths = run_inference(reid_model, query_data, args.device)

    # compute accuracy for dataset:
    compute_accuracy(q_feats, q_paths, g_feats, g_paths)


if __name__ == '__main__':
    args = get_args()
    usage_example_folder_path_query(args)
    # usage_example_track_as_query(args)

