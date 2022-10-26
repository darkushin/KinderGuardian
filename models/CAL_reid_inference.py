import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
import math

sys.path.append('Simple_CCReID')

from Simple_CCReID.models.img_resnet import ResNet50
from Simple_CCReID.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50
from Simple_CCReID.configs.default_img import get_img_config
from Simple_CCReID.configs.default_vid import get_vid_config
from Simple_CCReID.configs.default_img import _C as cfg
from Simple_CCReID.configs.default_vid import _C as cfg_vid
from Simple_CCReID.data.datasets.street42 import Street42
# from Simple_CCReID.data.dataloader import DataLoaderX
from Simple_CCReID.data.dataset_loader import VideoDataset
# from Simple_CCReID.data.samplers import DistributedInferenceSampler
from Simple_CCReID.test import extract_vid_feature

from centroids_reid.inference.inference_utils import ImageDataset, TrackDataset
import Simple_CCReID.data.img_transforms as T
import Simple_CCReID.data.spatial_transforms as ST
from torch.utils.data import DataLoader


__factory = {
    'resnet50': ResNet50,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}

VID_DATASET = ['ccvid', 'street42']

def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


def set_CAL_reid_cfgs(args):
    config = cfg.clone()
    config.merge_from_file(args.reid_config)
    config.merge_from_list(args.reid_opts)
    return config


def set_CAL_VID_reid_cfgs(args):
    config = cfg_vid.clone()
    config.merge_from_file(args.reid_config)
    config.merge_from_list(args.reid_opts)
    return config


def build_CAL_VID_gallery(config):
    dataset = Street42(root=config.DATA.ROOT, seq_len=config.AUG.SEQ_LEN, stride=config.AUG.SAMPLING_STRIDE)
    spatial_transform_test, temporal_transform_test = build_vid_transforms(config)
    galleryloader = DataLoader(
        dataset=VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test),
        # sampler=DistributedInferenceSampler(dataset.recombined_gallery),
        batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=False, shuffle=False)
    return dataset, galleryloader


def recombination_for_testset(dataset, seq_len=16, stride=4):
    ''' Split all videos in test set into lots of equilong clips.

    Args:
        dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
        seq_len (int): sequence length of each output clip
        stride (int): temporal sampling stride

    Returns:
        new_dataset (list): output dataset with lots of equilong clips
        vid2clip_index (list): a list contains the start and end clip index of each original video
    '''
    new_dataset = []
    vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
    for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
        # start index
        vid2clip_index[idx, 0] = len(new_dataset)
        # process the sequence that can be divisible by seq_len*stride
        for i in range(len(img_paths) // (seq_len * stride)):
            for j in range(stride):
                begin_idx = i * (seq_len * stride) + j
                end_idx = (i + 1) * (seq_len * stride)
                clip_paths = img_paths[begin_idx: end_idx: stride]
                assert (len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid, clothes_id))
        # process the remaining sequence that can't be divisible by seq_len*stride
        if len(img_paths) % (seq_len * stride) != 0:
            # reducing stride
            new_stride = (len(img_paths) % (seq_len * stride)) // seq_len
            for i in range(new_stride):
                begin_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + i
                end_idx = len(img_paths) // (seq_len * stride) * (seq_len * stride) + seq_len * new_stride
                clip_paths = img_paths[begin_idx: end_idx: new_stride]
                assert (len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len
            if len(img_paths) % seq_len != 0:
                clip_paths = img_paths[len(img_paths) // seq_len * seq_len:]
                # loop padding
                while len(clip_paths) < seq_len:
                    for index in clip_paths:
                        if len(clip_paths) >= seq_len:
                            break
                        clip_paths.append(index)
                assert (len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid, clothes_id))
        # end index
        vid2clip_index[idx, 1] = len(new_dataset)
        assert ((vid2clip_index[idx, 1] - vid2clip_index[idx, 0]) == math.ceil(len(img_paths) / seq_len))

    return new_dataset, vid2clip_index.tolist()



def CAL_build_model(config, device=None):
    """
    Build the CAL model according to the given config.
    """
    model = __factory[config.MODEL.NAME](config)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: Did not load a checkpoint for CAL model!")
    if device:
        model = model.to(device)
    model.eval()
    return model


def _inference(model, batch, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        imgs, _, filename = batch
        flip_imgs = torch.flip(imgs, [3])

        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)
    return batch_features, filename


def CAL_run_inference(model, dataloader, device=None):
    embeddings, paths = [], []
    if device:
        model = model.to(device)

    for x in dataloader:
        embedding, path = _inference(model, x, device)
        for vv, pp in zip(embedding, path):
            paths.append(pp)
            embeddings.append(vv.detach().cpu().numpy())

    embeddings = np.array(np.vstack(embeddings))
    paths = np.array(paths)
    return embeddings, paths


@torch.no_grad()
def extract_img_feature(model, dataloader, device='cuda:0'):
    features, pids, camids = [], torch.tensor([]), torch.tensor([])
    model.to(device)
    for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features_flip = model(flip_imgs)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)

        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
    features = torch.cat(features, 0)

    return features, pids, camids


def build_img_transforms(config):
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_test


def build_vid_transforms(config):
    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = None
    return spatial_transform_test, temporal_transform_test


def CAL_make_inference_data_loader(config, path, dataset_class, loaded_imgs=None):
    transform_test = build_img_transforms(config)

    num_workers = config.DATA.NUM_WORKERS
    if loaded_imgs:  # images are already loaded, use them with the custom TrackDataset dataloader
        val_set = dataset_class(loaded_imgs, transform_test)
    else:
        val_set = dataset_class(path, transform_test)
    val_loader = DataLoader(
        val_set,
        batch_size=config.DATA.TEST_BATCH,
        shuffle=False,
        num_workers=num_workers,
    )
    return val_loader


def CAL_track_inference(model, cfg, track_imgs, device):
    """
    Given loaded images of a track, run CTL inference on these images and return the embeddings of these images.
    """
    query_data = CAL_make_inference_data_loader(cfg, path='', dataset_class=TrackDataset, loaded_imgs=track_imgs)
    q_feats, _ = CAL_run_inference(model, query_data, device)
    return q_feats


if __name__ == '__main__':
    # Build configs
    config = parse_option()

    # Build model:
    model = CAL_build_model(config)
    model.eval()

    # compute feature vectors given a path to a folder:
    gallery_data = CAL_make_inference_data_loader(config, path='/home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0808_verified/bounding_box_test', dataset_class=ImageDataset)
    g_feats, g_paths = CAL_run_inference(model, gallery_data, device='cuda:0')
