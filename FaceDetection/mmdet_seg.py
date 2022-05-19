import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from FaceDetection.faceDetector import FaceDetector, is_img
import matplotlib.pyplot as plt
import numpy as np


def my_imshow(inp, labels=None):
    """Imshow for Tensor."""
    plt.figure()
    img = inp.permute(1, 2, 0).int().numpy()
    plt.imshow(img[:,:,::-1])
    plt.show()



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
    bbox_result, mask_result = np.array(result)[:, 0]
    ind = bbox_result[:, 4] >= 0.3
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
    return bboxes, masks


def detect_faces(img):
    """
    Given an img, run the face detector on it and return the bounding boxes of the faces in it and their probabilities.
    Bounding boxes are of the form [x0, y0, x1, y1] where (x0,y0) and (x1,y1) are the top-left and bottom right corners.
    """
    face_detector = FaceDetector(keep_all=True, device='cuda:1')
    bboxes, probs = face_detector.facenet_detecor.detect(img=img)
    return bboxes, probs


def create_segmentation(img, acc_th=0.3):
    """
    Given an img, create an instance segmentation of the persons that appear in the image and return the segmentation of
    the main person (largest segmentation) in the image.
    """
    # Initialize the detector model:
    config = mmcv.Config.fromfile("/home/bar_cohen/mmdetection/configs/solo/solo_r50_fpn_3x_coco.py")
    config.model.pretrained = False
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, "/home/bar_cohen/mmdetection/checkpoints/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth", map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()

    # run the model on the given image:
    result = inference_detector(model, img)
    assert isinstance(result, tuple)
    bbox_result, mask_result = np.array(result)[:, 0]

    # filter the results according to the given threshold:
    ind = bbox_result[:, 4] >= acc_th
    bbox_result = bbox_result[ind]

    # select the result with the largest segmentation (we assume the labeled person takes most of the image):
    max_bbox_ind = np.argmax(np.sum(np.array(mask_result)[ind], axis=(1, 2)))
    bboxes = bbox_result[max_bbox_ind]
    mask_result = mask_result[max_bbox_ind]
    x_any = mask_result.any(axis=0)
    y_any = mask_result.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    if len(x) > 0 and len(y) > 0:
        bboxes[0:4] = np.array(
            [x[0], y[0], x[-1] + 1, y[-1] + 1],
            dtype=np.float32)
    return bboxes, mask_result


def create_face_masks(face_bboxes, shape):
    face_masks = np.zeros((len(face_bboxes), shape[0], shape[1]), dtype=bool)
    for i in range(len(face_bboxes)):
        tl_y, tl_x, br_y, br_x = face_bboxes[i].astype(int)  # tl_x == top-left x coordinate, br_x == bottom_right
        face_masks[i][tl_x: br_x, tl_y: br_y] = True
    return face_masks


def find_matching_face(face_masks, segm_mask, alpha=0.5):
    """
    Given the face masks of all images in the img and the segmentation mask of the main person in the image, check if
    the face mask overlaps the segmentation mask
    """
    overlaps = []
    for face_mask in face_masks:
        face_size = np.sum(face_mask)
        overlap_area = np.logical_and(face_mask, segm_mask)
        overlap_size = np.sum(overlap_area)
        if overlap_size / face_size >= alpha:
            overlaps.append(True)
        else:
            overlaps.append(False)
    return overlaps


def plot_result(img, face_bboxes, segm_mask, matching_faces):
    fig = plt.figure(figsize=(25, 17))
    rows = 3
    columns = 1

    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.title('original image')
    fig.add_subplot(rows, columns, 1)

    plt.imshow(face_masks[0], cmap='gray')
    plt.axis('off')
    plt.title('face_mask')
    fig.add_subplot(rows, columns, 2)

    plt.imshow(segm_mask, cmap='gray')
    plt.axis('off')
    plt.title('segm_mask')
    fig.add_subplot(rows, columns, 3)

    # for i, face in enumerate(matching_faces):
    #     tl_y, tl_x, br_y, br_x = face_bboxes[i].astype(int)  # tl_x == top-left x coordinate, br_x == bottom_right
    #     face_im = img[tl_x: br_x, tl_y: br_y]
    #     plt.axis('off')
    #     if face:
    #         plt.title('Face matching segmentation')
    #     else:
    #         plt.title('Face and segmentation do not overlap')
    #     plt.imshow(face_im[:, :, ::-1])
    #     fig.add_subplot(rows, columns, 4)

    plt.show()


if __name__ == '__main__':

    device = "cuda:1"
    # load the image:
    img = '/home/bar_cohen/raid/OUR_DATASETS/10146_orig.jpg'
    # img = "/mnt/raid1/home/bar_cohen/OUR_DATASETS/10172_orig.jpg"
    if isinstance(img, str):
        img = mmcv.imread(img)

    # First check if there is a face in the image:
    face_bboxes, face_probs = detect_faces(img)
    face_bboxes[face_bboxes < 0] = 0

    # if at least one face was detected use segmentation
    if len(face_bboxes) > 0:
        plt.imshow(img[:, :, ::-1])
        plt.title('original image')
        plt.show()

        # create a mask of the face bbox:
        face_masks = create_face_masks(face_bboxes, img.shape[:2])  # take the shape without the channels
        plt.imshow(face_masks[0], cmap='gray')
        plt.title('face_mask')
        plt.show()

        # create a segmentation mask:
        segm_bbox, segm_mask = create_segmentation(img)
        plt.imshow(segm_mask, cmap='gray')
        plt.title('segm_mask')
        plt.show()

        # check which faces overlap:
        matching_faces = find_matching_face(face_masks, segm_mask)

        # plot_result(img, face_bboxes, segm_mask, matching_faces)
        # save overlapping faces:
        for i, face in enumerate(matching_faces):
            tl_y, tl_x, br_y, br_x = face_bboxes[i].astype(int)  # tl_x == top-left x coordinate, br_x == bottom_right
            face_im = img[tl_x: br_x, tl_y: br_y]
            plt.figure()
            if face:
                plt.title('Face matching segmentation')
            else:
                plt.title('Face and segmentation do not overlap')
            plt.imshow(face_im[:, :, ::-1])
            plt.show()

    # todo: if we found a matching face, we should extract it as before and not using the detected bbox!


# img = '/home/bar_cohen/raid/FaceData/labled_images/labled_again/Adam/10172_orig.jpg'
#
# # img = "/mnt/raid1/home/bar_cohen/OUR_DATASETS/10146_orig.jpg"
# if isinstance(img, str):
#     img = mmcv.imread(img)
# face_detector = FaceDetector(keep_all=True, device='cuda:1')
# # bboxes, probs = face_detector.facenet_detecor.detect(img=img)
# face_img, face_prob = face_detector.get_single_face(img, is_PIL_input=True)
# my_imshow(face_img)


# with torch.no_grad():
#     result = inference_detector(model, img)
#     show_result_pyplot(model, img, result, score_thr=0.3)
#     show_result(model, img, result, score_thr=0.3)
