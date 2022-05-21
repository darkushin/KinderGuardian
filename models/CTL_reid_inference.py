import sys
sys.path.append('centroids_reid')
from centroids_reid.inference.inference_utils import ImageDataset, TrackDataset, make_inference_data_loader, calculate_centroids, create_pid_path_index
from centroids_reid.train_ctl_model import CTLModel
from centroids_reid.config import cfg
from centroids_reid.utils.reid_metric import get_dist_func
from centroids_reid.datasets.transforms import ReidTransforms
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
    model.eval()
    with torch.no_grad():
        data, _, filename = batch
        _, global_feat = model.backbone(data.cuda(device=device))
        if normalize_with_bn:
            global_feat = model.bn(global_feat)
        return global_feat, filename


def run_inference(model, val_loader, device=None):
    embeddings = []
    paths = []
    if device:
        device = int(device.split(':')[1])
        model = model.cuda(device=device)

    for x in tqdm.tqdm(val_loader, total=len(val_loader)):
        embedding, path = _inference(model, x, device)
        for vv, pp in zip(embedding, path):
            paths.append(pp)
            embeddings.append(vv.detach().cpu().numpy())

    embeddings = np.array(np.vstack(embeddings))
    paths = np.array(paths)
    print('Finished creating CTL embeddings!')
    return embeddings, paths


def setup_CTL_reid(reid_cfg, ckpt):
    """
    Create a CTL ReID model from the given checkpoint.
    """
    model = CTLModel.load_from_checkpoint(ckpt)
    val_loader = make_inference_data_loader(reid_cfg, reid_cfg.DATASETS.ROOT_DIR, ImageDataset)
    if len(val_loader) == 0:
        raise RuntimeError("Lenght of dataloader = 0")
    return model, val_loader


def set_CTL_reid_cfgs(args):
    cfg.merge_from_file(args.reid_config)
    cfg.merge_from_list(args.reid_opts)
    return cfg


def create_gallery_features(model, data_loader, device, output_path=CTL_PICKLES):
    print('Starting to create gallery feature vectors')
    g_feat, paths_gallery = run_inference(model, data_loader, device)
    if cfg.MODEL.USE_CENTROIDS:
        pid_path_index = create_pid_path_index(paths=paths_gallery, func=extract_id)
        g_feat, paths_gallery = calculate_centroids(g_feat, pid_path_index)
        print('Created gallery feature vectors using centroids.')
    else:
        print('Did not use centroids for gallery feature vectors.')

    os.makedirs(output_path, exist_ok=True)
    print(f'Saving gallery feature vectors to: {output_path}')
    pickle.dump(g_feat, open(os.path.join(output_path, 'g_feats.pkl'), 'wb'))
    pickle.dump(paths_gallery, open(os.path.join(output_path, 'g_paths.pkl'), 'wb'))

    # todo: check if normalization should be applied
    # # normalize the feature vectors:
    # g_feat = torch.nn.functional.normalize(g_feat, dim=1, p=2)
    # device = torch.device(device)
    # g_feat = g_feat.to(device)

    return g_feat, paths_gallery


def load_gallery_features(gallery_path=CTL_PICKLES):
    print('Loading gallery feature vectors from pickles...')
    g_feat = pickle.load(open(os.path.join(gallery_path, 'g_feats.pkl'), 'rb'))
    paths_gallery = pickle.load(open(os.path.join(gallery_path, 'g_paths.pkl'), 'rb'))

    # g_feat = torch.from_numpy(np.load(os.path.join(gallery_path, 'g_feats.npy'), allow_pickle=True))
    # paths_gallery = np.load(os.path.join(gallery_path, 'g_paths.npy'), allow_pickle=True)

    # todo: check if necessary, I think it is already in when saving the feature vectors
    # # normalize the images:
    # g_feat = torch.nn.functional.normalize(g_feat, dim=1, p=2)
    # device = torch.device(device)
    # g_feat = g_feat.to(device)

    return g_feat, paths_gallery


def CTL_reid_dataset_inference(model, query_data, device):
    print('Computing query feature vectors')
    # track_batch = (track_images, '', '')
    # q_feat = _inference(model, track_batch, device, normalize_with_bn=False)  # we should not normalize with BN in this case
    q_feat, q_paths = run_inference(model, query_data, device)
    # todo: this needs to change to images and not dataloader, and images should be given as tensors


    # todo: check if normalization should be applied
    # # normalize the feature vectors:
    # q_feat = torch.nn.functional.normalize(q_feat, dim=1, p=2)
    # device = torch.device(device)
    # q_feat = q_feat.to(device)
    return q_feat, q_paths


def ctl_track_inference(model, cfg, track_imgs, device):
    query_data = make_inference_data_loader(cfg, path='', dataset_class=TrackDataset, loaded_imgs=track_imgs)
    q_feats, _ = CTL_reid_dataset_inference(model, query_data, device)
    return q_feats





# def create_batches(images, cfg):  # todo: need to add param of model.TEST.IMS_PER_BATCH
#     transforms_base = ReidTransforms(cfg)
#     val_transforms = transforms_base.build_transforms(is_train=False)
#
#     for ima



# def CTL_reid_track_inference(model, cfg, track_images, device):
#     print('Computing query feature vectors')
#     embeddings = []
#     # Apply the transforms defined in the cfg, and return all images in batches
#     track_batches = create_batches(cfg, track_images)
#     # track_images = torch.from_numpy(np.array(track_images).astype(np.int8))
#     # track_batch = (track_images, '', '')
#
#     for track_batch in track_batches:
#         q_feat, _ = _inference(model, track_batch, device)  # todo: check if we should normalize with BN in this case (by default we do)
#         for embedding in q_feat:
#             embeddings.append(embedding.detach().cpu().numpy())
#
#     # todo: check if normalization should be applied
#     # # normalize the feature vectors:
#     # q_feat = torch.nn.functional.normalize(q_feat, dim=1, p=2)
#     # device = torch.device(device)
#     # q_feat = q_feat.to(device)
#     return embeddings, _


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
    print('finished creating distmat')
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
    # g_feats, g_paths = create_gallery_features(reid_model, gallery_data, int(args.device.split(':')[1]), output_path=CTL_PICKLES)

    # OR load gallery feature:
    g_feats, g_paths = load_gallery_features(gallery_path=CTL_PICKLES)
    
    # # compute query feature vectors:
    query_imgs_path = '/home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0730/bounding_box_test'
    query_data = make_inference_data_loader(reid_cfg, query_imgs_path, ImageDataset)
    q_feats, q_paths = CTL_reid_dataset_inference(reid_model, query_data, args.device)

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
    # g_feats, g_paths = create_gallery_features(reid_model, gallery_data, int(args.device.split(':')[1]), output_path=CTL_PICKLES)

    # OR load gallery feature:
    g_feats, g_paths = load_gallery_features(gallery_path=CTL_PICKLES)

    # compute query feature vectors:
    track_imgs = pickle.load(open(os.path.join(CTL_PICKLES, 'track_imgs.pkl'), 'rb'))
    # q_feats, q_paths = CTL_reid_track_inference(reid_model, track_imgs, int(args.device.split(':')[1]))

    query_data = make_inference_data_loader(reid_cfg, path='', dataset_class=TrackDataset, loaded_imgs=track_imgs)
    q_feats, q_paths = CTL_reid_dataset_inference(reid_model, query_data, args.device)


    print('daniel')


    # todo: remove this block, it is just to speed up debugining:
    # print('Saving query feature vectors')
    # pickle.dump(q_feats, open(os.path.join(CTL_PICKLES, 'q_feats.pkl'), 'wb'))
    # pickle.dump(q_paths, open(os.path.join(CTL_PICKLES, 'q_paths.pkl'), 'wb'))
    # print('Loading query feature vectors')
    # q_feats = pickle.load(open(os.path.join(CTL_PICKLES, 'q_feats.pkl'), 'rb'))
    # q_paths = pickle.load(open(os.path.join(CTL_PICKLES, 'q_paths.pkl'), 'rb'))

    # create a probability vector for every query image:
    # reid_probs_dict = create_query_prob_vector(q_feats, g_feats, g_paths)


if __name__ == '__main__':
    args = get_args()
    usage_example_folder_path_query(args)
    # usage_example_track_as_query(args)

