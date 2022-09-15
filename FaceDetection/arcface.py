import copy
import pickle
import glob
import PIL
import mmcv
import numpy as np
import cv2
import os
import torch
from insightface.app import FaceAnalysis

from DataProcessing.dataProcessingConstants import ID_TO_NAME
import tqdm
GALLERY_PKL_PATH = "/mnt/raid1/home/bar_cohen/42street/pkls/gallery.pkl"
GPIDS_PKL_PATH = "/mnt/raid1/home/bar_cohen/42street/pkls/gpids.pkl"
# GALLERY_PKL_PATH_MIN_FACE_MARGIN = "/mnt/raid1/home/bar_cohen/42street/pkls/gallery_mmargin.pkl"
# GPIDS_PKL_PATH_MIN_FACE_MARGIN = "/mnt/raid1/home/bar_cohen/42street/pkls/gpids_mmargin.pkl"
GPIDS_NO_UNKNOWNS = "/mnt/raid1/home/bar_cohen/42street/pkls/gpids_no_unknowns.pkl"
GALLERY_NO_UNKNOWNS = "/mnt/raid1/home/bar_cohen/42street/pkls/gallery_no_unknowns.pkl"

model_path = '/home/bar_cohen/D-KinderGuardian/insightface_test/checkpoints/w600k_r50.onnx'
IMG_SIZE = ((112,112))
# CHAR_2_ID = {'main':1, 'blond':2, 'guy2': 4, 'w1':5, 'w2':6, 'young_guy':7, 'mustache':8, 'young_w3':9}
# ID_2_NAME = {v:k for k, v in CHAR_2_ID.items()}


class ArcFace():

    def __init__(self,gallery_path=None, device=1):
        self.model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=device)  # given gpu id, if negative, then use cpu
        self.face_recognition = self.model.models['recognition']
        self.gallery_path = gallery_path
        self.gallery = None # call read_gallery to set
        self.gpids = None

    def get_img_embedding_from_file(self,file):
        img = cv2.imread(file)
        img = cv2.resize(img, IMG_SIZE)
        embedding = self.face_recognition.get_feat(img)
        return embedding

    def read_gallery_from_pkl(self, gallery_path:str, gpid_path:str):
        if os.path.isfile(gallery_path) and os.path.isfile(gpid_path):
            with open(gallery_path, 'rb') as f:
                self.gallery = pickle.load(f)
                f.close()

            with open(gpid_path, 'rb') as f:
                self.gpids = pickle.load(f)
                f.close()
        else:
            raise "Entered Gallery or GPID paths are incorrect."
        print("Done Loading Gallery from pkl paths")

    def read_gallery_from_scratch(self):
        print("Creating gallery from scratch...")
        embeddings = []
        gpids = []
        folders = [os.path.join(self.gallery_path, file) for file in os.listdir(self.gallery_path)]
        for folder in folders:
            if os.path.isdir(folder):
                files = [os.path.join(self.gallery_path,folder, file) for file in os.listdir(os.path.join(self.gallery_path, folder))]
                embeddings.extend([self.get_img_embedding_from_file(file) for file in files])
                cur_id = os.path.join(self.gallery_path,folder).split('/')[-1]
                gpids.extend([cur_id] * len(files))

        self.gallery = np.array(embeddings)
        self.gpids = np.array(gpids)

    def create_feats(self, path):
        """
        Create feature vectors for all images in the given path.
        """
        embeddings = []
        files = [os.path.join(path, file) for file in os.listdir(path) if '.png' in file]
        embeddings.extend([self.get_img_embedding_from_file(file) for file in files])
        return np.array(embeddings)

    def save_gallery_to_pkl(self, gallery_path:str , gpid_path:str):
        if self.gallery is not None and self.gpids is not None:
            with open(gallery_path, 'wb') as f:
                pickle.dump(self.gallery,f)
                f.close()

            with open(gpid_path, 'wb') as f:
                pickle.dump(self.gpids, f)
                f.close()

    def img_tensor_to_cv2(self, face):
        numpy_img = face.permute(1, 2, 0).int().numpy().astype(np.uint8)
        face = numpy_img[:, :, ::-1]
        return face

    def predict_img(self, img):
        if type(img) == torch.Tensor:
            img = self.img_tensor_to_cv2(img)
        if img.shape != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)
        input_feat = self.face_recognition.get_feat(img)
        scores = {i:0 for i in ID_TO_NAME.keys()}
        for i in scores.keys():
            gallery_of_i = self.gallery[self.gpids == ID_TO_NAME[i]]
            if gallery_of_i is not None and len(gallery_of_i) > 0:
                cur_sims = [self.face_recognition.compute_sim(input_feat, cand) for cand in gallery_of_i]
                mean_scores = np.mean(cur_sims)
                # max_scores = np.max(cur_sims)
                # top_5_mean_score = np.argpartition(cur_sims, 5)[-5:].mean()
                # print(ID_TO_NAME[i], "mean score:", mean_scores, "max score", max_scores)
                # scores[i] = mean_scores if mean_scores > self.score_threshold else 0
                # scores[i] = top_5_mean_score
                scores[i] = mean_scores if mean_scores > 0.25 else 0
                # if scores[i] > 0:
                    # print(f"Gotcha, {i}")
        return scores

    def vectorized_cosine_sim_compare(self, input_feat, gallery_of_id_i):



        # input_feat = input_feat.ravel()
        pass



    def predict_track(self, imgs:np.array):
        scores = {i:0 for i in ID_TO_NAME.keys()}
        for img in imgs:
            cur_img_scores = self.predict_img(img)
            for k in scores.keys():
                # taking the max score across the track
                scores[k] += cur_img_scores[k]  #These are unbalanced scores
                # if cur_img_scores[k] > scores[k]:
                #     scores[k] = cur_img_scores[k]
        return scores


    def predict_track_vectorized(self, imgs:np.array):
        # converted_imgs_to_np = np.zeros_like(imgs.shape)
        # for i, img in enumerate(imgs):
        #     if type(img) == torch.Tensor:
        #         img = self.img_tensor_to_cv2(img)
        #     if img.shape != IMG_SIZE:
        #         img = cv2.resize(img, IMG_SIZE)
        #     converted_imgs_to_np[i] = img
        import copy
        resized_imgs = np.array([cv2.resize(img, IMG_SIZE) for img in imgs])
        imgs_feats_tensor = torch.tensor([self.face_recognition.get_feat(img) for img in resized_imgs]).squeeze()
        scores = {i: 0 for i in ID_TO_NAME.keys()}
        cosineSim = torch.nn.CosineSimilarity(dim=1)
        for i in scores.keys():
            cur_img_feats_tensor = copy.deepcopy(imgs_feats_tensor)
            gallery_of_i = self.gallery[self.gpids == ID_TO_NAME[i]]
            if len(gallery_of_i) > 0:  # this gallery is not empty
                gallery_of_i_tensor = torch.tensor(gallery_of_i).squeeze()
                if gallery_of_i_tensor.size()[0] < cur_img_feats_tensor.size()[0]:  # pad gallery
                    temp = torch.zeros_like(cur_img_feats_tensor)
                    temp[:gallery_of_i_tensor.size()[0], :] = gallery_of_i_tensor
                    gallery_of_i_tensor = temp
                else:  # pad img_feats
                    temp = torch.zeros_like(gallery_of_i_tensor)
                    temp[:imgs_feats_tensor.size()[0], :] = imgs_feats_tensor
                    cur_img_feats_tensor = temp
                cur_cos_sim = cosineSim(cur_img_feats_tensor, gallery_of_i_tensor)
                scores[i] = float(cur_cos_sim.mean())
        return scores





        # for i in
        # normalize scores
        # np_ids_score = np.array(list(scores.values()))
        # mean_score = np.mean(np_ids_score)
        # std_score = np.std(np_ids_score)
        #
        # # max_score = max(scores.values())
        # # if max_score > 0:
        # for k, v in scores.items():
        #     scores[k] = (v - mean_score) / (std_score + 0.000001)

        return scores



    def detect_face_from_img(self, crop_img, threshold=0):
        """
        Given an img file, detect the face in the image and return it. Returns None if no face image was
        detected.
        :param img_path: the img file to detect.
        """

        # ToDo deal with multiface in a single crop, currently only a single face will be detected
        face_imgs = []
        face_bboxs = []
        face_probs = []

        detection_res = self.model.get(crop_img)
        for i in range(len(detection_res)):
            if len(detection_res) > 0 and detection_res[i] and detection_res[i]['det_score'] >= threshold:
                face_bbox = detection_res[i]['bbox']
                X = np.max([int(face_bbox[0]), 0])
                Y = np.max([int(face_bbox[1]), 0])
                W = np.min([int(face_bbox[2]), crop_img.shape[1]])
                H = np.min([int(face_bbox[3]), crop_img.shape[0]])
                face_imgs.append(crop_img[Y:H, X:W])
                face_bboxs.append(face_bbox)
                face_probs.append(detection_res[i]['det_score'])

        return face_imgs, face_bboxs , face_probs

    def detect_face_from_file_crop(self, img_path, threshold=0):
        """
        Given a path to an image, detect the face in the image and return it. Returns None if no face image was
        detected.
        :param img_path: the path to the image in which a face should be detected.
        """
        crop_img = cv2.imread(img_path)
        face_img = self.detect_face_from_img(crop_img=crop_img, threshold=threshold)
        return face_img

def is_img(img):

    return img is not None

def collect_faces_from_video(video_path:str, face_dir:str, skip_every=1, face_det_threshold = 0.8):
    os.makedirs(face_dir, exist_ok=True)
    arc = ArcFace()
    vid_name = video_path.split('/')[-1]
    imgs = mmcv.VideoReader(video_path)
    saved_faces = 0
    # trans = transforms.ToPILImage()
    for i ,img in enumerate(tqdm.tqdm(imgs)):
        if i % skip_every == 0:
            face_imgs, face_bboxes, face_probs = arc.detect_face_from_img(crop_img=img)
            if face_imgs is not None:
                for face, prob in zip(face_imgs, face_probs):
                    if prob >= face_det_threshold:
                        face = face[:, :, ::-1]
                        PIL_img = PIL.Image.fromarray(face).convert('RGB')
                        PIL_img.save(os.path.join(face_dir, f'{vid_name}_{i}.png'))
                        saved_faces += 1
                    # img_show = face.permute(1, 2, 0).int().numpy().astype(np.uint8)
                    # img_show = img_show[:,:,::-1]
                    # plt.clf()
                    # # # plt.title(f'Detected Face, label:  {ID_TO_NAME[id]}, counter : {counter}')
                    # plt.imsave(os.path.join(face_dir, f'{vid_name}_{i}.png'), img_show)
    print(f'Done running video {video_path}, total faces saved: {saved_faces}')


def collect_faces_from_list_of_videos(list_of_videos:list,face_dir:str,skip_every=1, face_det_threshold= 0.8):
    for video_path in list_of_videos:
        print(video_path)
        collect_faces_from_video(video_path=video_path, face_dir=face_dir, skip_every=skip_every, face_det_threshold= face_det_threshold)

def create_clusters(k,face_crops_path, cluster_path, method='arcface_feats'):
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
    if method == 'raw_images':
        print('Creating clusters from raw images.')
        images = np.array(np.float32(images).reshape(len(images), -1) / 255)
        model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        pred_images = model.predict(images.reshape(-1, 224, 224, 3))
    elif method == 'arcface_feats':
        print('Creating clusters from arcface feature vectors.')
        arc = ArcFace()
        pred_images = arc.create_feats(face_crops_path)
    else:
        raise Exception('Please specify a method for clustering. Options: [raw_images, arcface_feats]')
    pred_images = pred_images.reshape(pred_images.shape[0], -1)
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
    videos_path = "/mnt/raid1/home/bar_cohen/42street/val_videos_1/"
    clusters = "/mnt/raid1/home/bar_cohen/42street/face_part1_face_clusters_min_margin_no_skip_low_thresh_new_face_det/"
    faces = '/mnt/raid1/home/bar_cohen/42street/face_part1_no_skip_low_thresh_new_face_det/'

    vids_list = [os.path.join(videos_path, f) for f in os.listdir(videos_path)]
    collect_faces_from_list_of_videos(vids_list,face_dir=faces, skip_every=1)
    create_clusters(k=100,cluster_path=clusters,face_crops_path=faces)
if __name__ == '__main__':
    main()



# if __name__ == '__main__':
#     gpath = "/mnt/raid1/home/bar_cohen/42street/corrected_face_clusters/"

    # arc = ArcFace(gpath)
    # img1_filepath = "/mnt/raid1/home/bar_cohen/42street/clusters/mustache/_s31000_e31501.mp4_75.png"
    # img2_filepath = "/mnt/raid1/home/bar_cohen/42street/_s11000_e11501.mp4_200.png"
    #
    # img1 = cv2.imread(img1_filepath)
    # img2 = cv2.imread(img2_filepath)
    # both = np.array([img1,img2])
    # # arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH_MIN_FACE_MARGIN, gpid_path=GPIDS_PKL_PATH_MIN_FACE_MARGIN)
    # # arc.model.forward(both)
    # # arc.predict(both)
    #
    # arc.read_gallery_from_scratch()
    # arc.save_gallery_to_pkl(GALLERY_PKL_PATH_MIN_FACE_MARGIN, GPIDS_PKL_PATH_MIN_FACE_MARGIN)
    # print(arc.gallery)
    # arc.read_gallery_from_pkl(gallery_path=GALLERY_PKL_PATH, gpid_path=GPIDS_PKL_PATH)










