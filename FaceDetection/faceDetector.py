import glob
import os
import pickle
from collections import defaultdict

import cv2
import mmcv
import tqdm
from PIL import Image
from torchvision import transforms

from DataProcessing.DB.dal import get_entries, Crop
from DataProcessing.dataProcessingConstants import ID_TO_NAME, NAME_TO_ID
from FaceDetection.augmentions import normalize_image, crop_top_third_and_sides
from FaceDetection.facenet_pytorch import MTCNN
import numpy as np
from matplotlib import pyplot as plt


class FaceDetector():
    """
    The faceDetector class is in charge of using the MTCNN face detector
    params:
    raw_images_path - path to folder containing images for face detection
    face_data_path - path to folder already containing face images. This will be used in order to put said images into
    the correct format.
    """
    def __init__(self, faces_data_path:str=None, thresholds=[0.8,0.8,0.8],
                 keep_all=False, device='cuda:1',min_face_size=20):
        self.device = device
        self.faces_data_path = faces_data_path
        self.keep_all = keep_all
        self.facenet_detecor = MTCNN(margin=40, select_largest=True, post_process=False, device=device,
                                     keep_all=self.keep_all, thresholds=thresholds, min_face_size=min_face_size)

    def detect_single_face(self, img):
        # use face detector to find a single face in an image, rests to entered init of keeping face after change
        self.facenet_detecor.keep_all = False
        ret, prob, batch_boxes = self.facenet_detecor(img, return_prob=True)
        self.facenet_detecor.keep_all = self.keep_all
        return ret , prob, batch_boxes

    def get_single_face(self,img, is_PIL_input, norm=True):
        # TODO this is a horrible function and flow. will later be upgrades with instace segmentaion?
        face_img, prob, batch_boxes = self.facenet_detecor(img, return_prob=True)
        if is_img(face_img):
            if face_img.size()[0] > 1:  # two or more faces detected in the img crop
                # faceClassifer.imshow(face_img[0:2])
                face_img = crop_top_third_and_sides(img, is_PIL_input)
                if is_PIL_input:
                    face_img, prob, batch_boxes = self.detect_single_face(face_img)  # this returns a single img of dim 3
                    if is_img(face_img) and norm:
                        face_img = normalize_image(face_img)
                else:
                    # face_img, prob = self.detect_single_face_inv_norm(face_img)
                    face_img, prob = None, 0
                    # TODO try and view these images, make sure they are okay, are they normalized?
                    # if self.is_img(face_img):
                    #     to_show_copy = face_img
                    #     to_show_copy.permute(2, 1, 0).int().numpy()
                    #     plt.imshow(to_show_copy)
                    #     plt.show()
            else:
                face_img = face_img[0]  # current face_img shape is 1ximage size(dim=3), we only want the img itself
                face_img = normalize_image(face_img) if norm else face_img
                prob = prob[0]
        return face_img , prob, batch_boxes


    def filter_out_non_face_corps(self, recreate_data, save_images_path='') -> dict:
        """ Given a set of image crop filter out all images without the faces present
        and update the high conf face img member """

        high_conf_face_imgs = defaultdict(list)
        if self.faces_data_path and not recreate_data:
            print("pickle path to images received, loading...")
            high_conf_face_imgs = pickle.load(open(os.path.join(self.faces_data_path, 'images_with_crop_skip_5.pkl'),'rb'))
        else:
            norm = not save_images_path
            face_crops = get_entries(filters={Crop.is_face == True,
                                              Crop.reviewed_one == True,
                                              Crop.is_vague == False,
                                              Crop.invalid == False}).all()
            crops_path = "/mnt/raid1/home/bar_cohen/"
            raw_imgs_dict = defaultdict(list)
            for i, crop in tqdm.tqdm(enumerate(face_crops), total=len(face_crops)):
                if not i % 5 == 0:
                    continue
                # print(i ,'/',len(face_crops))
                # fix for v, v_ issue
                name = crop.im_name
                if not os.path.isfile(os.path.join(crops_path, crop.vid_name, name)):
                    name = 'v'+crop.im_name[2:]
                img = Image.open(os.path.join(crops_path, crop.vid_name, name))
                raw_imgs_dict[NAME_TO_ID[crop.label]].append((img.copy(),crop))
                img.close()

            print('Number of unique ids', len(raw_imgs_dict.keys()))
            given_num_of_images = sum([len(raw_imgs_dict[i]) for i in raw_imgs_dict.keys()])
            print(f"Received a total of {given_num_of_images} images")
            counter = 0
            errors = 0
            for id in raw_imgs_dict.keys():
                print('cur id is', id)
                for img,crop in raw_imgs_dict[id]:
                    print(f'{counter}/{given_num_of_images}')
                    if save_images_path:
                        pass
                        # plt.title(f'Original Crop, label: {ID_TO_NAME[id]} ,counter : {counter}')
                        # plt.imsave(os.path.join(to_save,str(ID_TO_NAME[id]),str(counter) + '_orig.jpg'),
                        #            np.array(img).astype(np.uint8))
                    try:
                        ret, _ = self.get_single_face(img, is_PIL_input=True, norm=norm)
                        if is_img(ret):
                            if save_images_path:
                                os.makedirs(os.path.join(save_images_path, str(ID_TO_NAME[id])), exist_ok=True)
                                img_show = ret.permute(1, 2, 0).int().numpy().astype(np.uint8)
                                plt.clf()
                                # plt.title(f'Detected Face, label:  {ID_TO_NAME[id]}, counter : {counter}')
                                plt.imsave(os.path.join(save_images_path, str(ID_TO_NAME[id]), f'{crop.im_name}'), img_show)
                            high_conf_face_imgs[id].append((ret,crop))
                        counter += 1
                    except Exception as e:
                        print(e)
                        errors += 1

            given_num_of_images_final = sum([len(high_conf_face_imgs[i]) for i in raw_imgs_dict.keys()])
            print(f'Post filter left with {given_num_of_images_final}')
            print(f'errors: {errors}')
            pickle.dump(high_conf_face_imgs, open(os.path.join('/mnt/raid1/home/bar_cohen/FaceData/', 'images_with_crop_skip_5.pkl'),'wb'))
        return high_conf_face_imgs

def is_img(img):
    return img is not None and img is not img.numel()


def collect_faces_from_video(video_path:str, face_dir:str, skip_every=25):
    fd = FaceDetector(faces_data_path=None,thresholds=[0.97,0.97,0.97],keep_all=True, min_face_size=45)
    vid_name = video_path.split('/')[-1]
    imgs = mmcv.VideoReader(video_path)
    trans = transforms.ToPILImage()
    for i ,img in enumerate(tqdm.tqdm(imgs)):
        if i % skip_every == 0:
            face_imgs = fd.facenet_detecor(img, return_prob=False)
            if is_img(face_imgs):
                for face in face_imgs:
                    numpy_img = face.permute(1, 2, 0).int().numpy().astype(np.uint8)
                    numpy_img = numpy_img[:, :, ::-1]
                    PIL_img = Image.fromarray(numpy_img).convert('RGB')
                    PIL_img.save(os.path.join(face_dir, f'{vid_name}_{i}.png'))
                    # img_show = face.permute(1, 2, 0).int().numpy().astype(np.uint8)
                    # img_show = img_show[:,:,::-1]
                    # plt.clf()
                    # # # plt.title(f'Detected Face, label:  {ID_TO_NAME[id]}, counter : {counter}')
                    # plt.imsave(os.path.join(face_dir, f'{vid_name}_{i}.png'), img_show)

def collect_faces_from_list_of_videos(list_of_videos:list,face_dir:str):
    for video_path in list_of_videos:
        collect_faces_from_video(video_path=video_path, face_dir=face_dir)

def create_clusters(k,face_crops_path, cluster_path):
    """
    Based on crops saved on the output path folder, run clustering to unsupervised-ly label the Ids
    @param k: K clusters to create
    @return: None
    """
    # we can extend this later to receive input from args
    # if generating the crops in the same run, the input for the clustering should be the output arg
    import tensorflow as tf
    import shutil

    os.makedirs(cluster_path, exist_ok=True)
    joined_path_to_files = os.path.join(face_crops_path, '*.*')
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(joined_path_to_files)]
    paths = [file for file in glob.glob(joined_path_to_files)]
    assert images and paths, "crops folder must be non-empty"
    images = np.array(np.float32(images).reshape(len(images), -1) / 255)
    model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    predictions = model.predict(images.reshape(-1, 224, 224, 3))
    pred_images = predictions.reshape(images.shape[0], -1)
    from sklearn.cluster import KMeans
    kmodel = KMeans(n_clusters=k, random_state=728)
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    for i in range(k):
        os.makedirs(f'{cluster_path}/cluster{str(i)}', exist_ok=True)
    for i in range(len(paths)):
        shutil.copy2(paths[i], f'{cluster_path}/cluster{str(kpredictions[i])}')

def main():
    """Simple test of FaceDetector"""
    videos_path = '/mnt/raid1/home/bar_cohen/42street/training_videos_part2'
    clusters = "/mnt/raid1/home/bar_cohen/42street/face_part2/clusters/"
    faces = '/mnt/raid1/home/bar_cohen/42street/face_part2/'

    vids_list = [os.path.join(videos_path, f) for f in os.listdir(videos_path)]
    collect_faces_from_list_of_videos(vids_list,'/mnt/raid1/home/bar_cohen/42street/face_part2/')
    # create_clusters(k=10,cluster_path=clusters,face_crops_path=faces)
    # fd.filter_out_non_face_corps(recreate_data=True, save_images=False),
    # from faceClassifer import load_data
    # le, dl_train, dl_val, dl_test =  load_data('/mnt/raid1/home/bar_cohen/FaceData/')
    # fd.build_samples_hist(le, dl_train, title='dataloader train')
    # fd.build_samples_hist(le, dl_val, title='dataloader val')
    # fd.build_samples_hist(le, dl_test, title='dataloader test')
    # X,y, _, _,_,_ = fd.create_X_y_faces() # only taining
    # print(len(X), len(y))
    # fd.build_samples_hist(fd.high_conf_face_imgs, 'Face Images Hist')
    # fd.build_samples_hist(read_labled_croped_images(fd.raw_images_path), title='Raw Images Hist')
if __name__ == '__main__':
    main()

