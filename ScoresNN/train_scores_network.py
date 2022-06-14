import sys

from torch import nn

sys.path.append('fast-reid')
sys.path.append('centroids_reid')
sys.path.append('models')

from models.model_runner import get_args
from models.CTL_reid_inference import *
from FaceDetection.faceClassifer import FaceClassifer
from FaceDetection.faceDetector import FaceDetector, is_img
from PIL import Image
from centroids_reid.datasets.transforms import ReidTransforms
from ScoresNN.scores_model import FACE_PREDICTION_SIZE, FACE_MIN_PROB, create_data_loader, ScoresNet, inference, \
    ReIDClassifier, train, evaluate
import wandb


GALLERY_PKL = "/mnt/raid1/home/bar_cohen/Scores_NN/"


def init_reid_model(reid_cfg):
    checkpoint = torch.load(reid_cfg.TEST.WEIGHT)
    checkpoint['hyper_parameters']['MODEL']['PRETRAIN_PATH'] = './centroids_reid/models/resnet50-19c8e357.pth'
    reid_model = CTLModel._load_model_state(checkpoint)
    return reid_model


def create_reid_training_features(args, reid_cfg, recreate=True):
    if not os.path.isdir(GALLERY_PKL) or recreate:
        os.makedirs(GALLERY_PKL, exist_ok=True)
        gallery_data = make_inference_data_loader(reid_cfg, reid_cfg.DATASETS.ROOT_DIR, ImageDataset)
        g_feats, g_paths = create_gallery_features(reid_model, gallery_data, args.device, output_path=GALLERY_PKL)
    else:
        g_feats, g_paths = load_gallery_features(gallery_path=CTL_PICKLES)

    g_feats = torch.from_numpy(g_feats)
    return g_feats, g_paths


def create_face_classifier():
    with open("/mnt/raid1/home/bar_cohen/FaceData/le_19.pkl", 'rb') as f:
        le = pickle.load(f)
    faceClassifer = FaceClassifer(num_classes=19, label_encoder=le, device='cuda:0')
    faceClassifer.model_ft.load_state_dict(torch.load("/mnt/raid1/home/bar_cohen/FaceData/checkpoints/REAPEAT OLD EXP 19, 0.pth"))
    faceClassifer.model_ft.eval()
    return faceClassifer


def create_face_training_predictions(imgs_path, face_detector, face_classifier):
    all_imgs_paths = os.listdir(imgs_path)
    # face_predictions = np.zeros((len(all_imgs_paths), FACE_PREDEICTION_SIZE))
    face_imgs = []
    face_probs = []
    for i, im_path in enumerate(all_imgs_paths):
        if i == 100:
            break
        crop_im = Image.open(os.path.join(imgs_path, im_path))
        face_img, face_prob = face_detector.get_single_face(crop_im, is_PIL_input=True)
        face_prob = face_prob if face_prob else 0
        face_imgs.append(face_img)
        face_probs.append(face_prob)

        # if face_prob > 0.5:  # todo: take only images with high probability
    face_pred, face_clf_outputs = face_classifier.predict(torch.stack(face_imgs))

    return face_clf_outputs

# create ReID inference model

# create feature embeddings for all the images in the gallery (training set)

# create face predictions for every image in the gallery

# apply face detector for all images and create a binary map of size(gallery) to indicate if an image has face or not

# initialize scores NN

# train scores NN

# eval scores NN - for every query image, get reid embedding and face prediction, send to scores NN and measure accuracy



def create_embeddings(path, reid_model, face_detector, face_classifier, device):
    """
    Given a path to the images that should be used for creating embeddings, create reid_embeddings and face predictions
    for each image in the folder.
    """
    embeddings = {}
    reid_model = reid_model.cuda(device=int(device.split(':')[1]))
    reid_model.eval()

    transforms_base = ReidTransforms(cfg)
    reid_transforms = transforms_base.build_transforms(is_train=False)
    for i, im_path in enumerate(os.listdir(path)):
        im_dict = {}
        crop_im = Image.open(os.path.join(path, im_path))
        transformed_img = reid_transforms(crop_im)[None, :]  # add another dimension as the reid_model expects batches
        with torch.no_grad():
            _, reid_feat = reid_model.backbone(transformed_img.cuda(device))
        reid_feat = reid_model.bn(reid_feat)
        im_dict['reid_embedding'] = reid_feat

        face_img, face_prob = face_detector.get_single_face(crop_im, is_PIL_input=True)
        face_prob = face_prob if face_prob else 0
        face_scores = torch.zeros(FACE_PREDICTION_SIZE)
        if face_prob >= FACE_MIN_PROB and is_img(face_img):
            face_imgs = [face_img, torch.zeros(face_img.shape)]  # todo: find other solution. workaround as the predict func expects multiple tensors and not a single one
            face_pred, face_scores = face_classifier.predict(torch.stack(face_imgs))
            face_scores = face_scores[0]
        im_dict['face_scores'] = face_scores

        embeddings[im_path] = im_dict
    return embeddings


if __name__ == '__main__':
    args = get_args()
    reid_cfg = set_CTL_reid_cfgs(args)

    reid_model = init_reid_model(reid_cfg)

    # reid_embeddings = create_reid_training_features(args, reid_cfg)

    # face_classifier = create_face_classifier()

    # face_detector = FaceDetector(keep_all=True, device=args.device)

    # face_predictions = create_face_training_predictions(reid_cfg.DATASETS.ROOT_DIR, face_detector, face_classifier)

    # embeddings = create_embeddings(path=reid_cfg.DATASETS.ROOT_DIR, reid_model=reid_model, face_detector=face_detector,
    #                                face_classifier=face_classifier, device=args.device)

    # train_loader = create_data_loader(reid_cfg, reid_model=reid_model, face_detector=face_detector,
    #                                   face_classifier=face_classifier, device=args.device)

    # scores_model = ScoresNet()

    # predictions = inference(scores_model, train_loader, device=args.device)

    wandb.init(project="ReID-clf-model", entity="kinder-guardian", config=cfg)

    ReID_classifier = ReIDClassifier()

    # train_loader = create_data_loader(reid_cfg, reid_model=reid_model, device=args.device, dataloader_type='ReIDClassifierDataset')

    # train(ReID_classifier, train_loader, num_epochs=400, criterion=nn.CrossEntropyLoss(),
    #       optimizer=torch.optim.Adam(ReID_classifier.parameters(), lr=1e-1), device=args.device)

    eval_loader = create_data_loader(reid_cfg, reid_model=reid_model, device=args.device, dataloader_type='ReIDClassifierDataset')
    evaluate(ReID_classifier, eval_loader, device=args.device)
