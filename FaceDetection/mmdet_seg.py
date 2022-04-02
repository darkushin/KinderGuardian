import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = "/home/bar_cohen/mmdetection/configs/solo/solo_r50_fpn_3x_coco.py"
# Setup a checkpoint file to load
checkpoint = "/home/bar_cohen/mmdetection/checkpoints/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth"

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = False

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()


def show_result(self,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):



    assert isinstance(result, tuple)
    bbox_result, mask_result = np.array(result)[:,0]
    ind = bbox_result[:,4] >= 0.3
    bbox_result = bbox_result[ind]
    max_bbox_ind = np.argmax(np.sum(np.array(mask_result)[ind], axis=(1, 2)))
    bboxes = bbox_result[max_bbox_ind]
    mask_result = mask_result[max_bbox_ind]
    masks = mask_result
    x_any = masks.any(axis=0)
    y_any = masks.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    if len(x) > 0 and len(y) > 0:
        bboxes[0:4] = np.array(
            [x[0], y[0], x[-1] + 1, y[-1] + 1],
            dtype=np.float32)
    return bboxes ,masks


img = "/mnt/raid1/home/bar_cohen/OUR_DATASETS/10146_orig.jpg"
import numpy as np
with torch.no_grad():
    result = inference_detector(model, img)
    # show_result_pyplot(model, img, result, score_thr=0.3)
    show_result(model, img, result, score_thr=0.3)

