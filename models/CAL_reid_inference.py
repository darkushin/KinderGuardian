import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('Simple_CCReID')

from Simple_CCReID.models.img_resnet import ResNet50
from Simple_CCReID.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50
from Simple_CCReID.configs.default_img import get_img_config
# from Simple_CCReID.configs.default_vid import get_vid_config
from centroids_reid.inference.inference_utils import ImageDataset, TrackDataset
import Simple_CCReID.data.img_transforms as T
from torch.utils.data import DataLoader


__factory = {
    'resnet50': ResNet50,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


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
    # if args.dataset in VID_DATASET:
    #     config = get_vid_config(args)
    # else:
    config = get_img_config(args)

    return config


def build_model(config):
    """
    Build the CAL model according to the given config.
    """
    model = __factory[config.MODEL.NAME](config)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
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


if __name__ == '__main__':
    # Build configs
    config = parse_option()

    # Build model:
    model = build_model(config)
    model.eval()

    # compute feature vectors given a path to a folder:
    gallery_data = CAL_make_inference_data_loader(config, path='/home/bar_cohen/KinderGuardian/fast-reid/datasets/same_day_0808_verified/bounding_box_test', dataset_class=ImageDataset)
    g_feats, g_paths = CAL_run_inference(model, gallery_data, device='cuda:0')
