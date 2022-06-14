import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tqdm

from centroids_reid.inference.inference_utils import pil_loader, get_all_images
from FaceDetection.faceDetector import is_img
from centroids_reid.datasets.transforms import ReidTransforms
import wandb


REID_EMBEDDING_SIZE = 2048
FACE_PREDICTION_SIZE = 19
FACE_MIN_PROB = 0.98
OUTPUT_PREDICTION_SIZE = 21
CHECKPOINTS_DIR = "/home/bar_cohen/raid/Scores_NN/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'reid_classifier_epoch_400.ckpt')


class ReIDClassifierDataset(Dataset):
    def __init__(self, dataset_path: str, reid_transform=None, loader=pil_loader, reid_model=None, device=None):
        self.dataset = get_all_images(dataset_path)
        self.transform = reid_transform
        self.loader = loader
        self.reid_model = reid_model
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.loader(img_path)
        label = int(os.path.basename(os.path.normpath(img_path)).split('_')[0]) - 1
        # label = torch.zeros(OUTPUT_PREDICTION_SIZE)
        # label[int(label_str) - 1] = 1

        if self.transform is not None:
            img = self.transform(img)

        if self.device:
            self.reid_model = self.reid_model.cuda(device=int(self.device.split(':')[1]))  # todo: check if possible to change to to(device)

        # create reid feature embedding:
        self.reid_model.eval()
        with torch.no_grad():
            _, reid_feat = self.reid_model.backbone(img[None, :].cuda(self.device))  # add another dimension as the reid_model expects batches
        reid_feat = self.reid_model.bn(reid_feat)[0]

        return {'features': reid_feat, 'label': label}


class ImageDataset(Dataset):
    def __init__(self, dataset_path: str, reid_transform=None, loader=pil_loader, reid_model=None, face_classifier=None,
                 face_detector=None, device=None):
        self.dataset = get_all_images(dataset_path)
        self.transform = reid_transform
        self.loader = loader
        self.reid_model = reid_model
        self.face_classifier = face_classifier
        self.face_detector = face_detector
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.loader(img_path)

        if self.transform is not None:
            reid_img = self.transform(img)

        if self.device:
            self.reid_model = self.reid_model.cuda(device=int(self.device.split(':')[1]))

        # create reid feature embedding:
        self.reid_model.eval()
        with torch.no_grad():
            _, reid_feat = self.reid_model.backbone(reid_img[None, :].cuda(self.device))  # add another dimension as the reid_model expects batches
        reid_feat = self.reid_model.bn(reid_feat)[0]

        # create face classification vector:
        face_img, face_prob = self.face_detector.get_single_face(img, is_PIL_input=True)
        face_prob = face_prob if face_prob else 0
        face_scores = torch.zeros(FACE_PREDICTION_SIZE).cuda(self.device)
        if face_prob >= FACE_MIN_PROB and is_img(face_img):
            face_imgs = [face_img, torch.zeros(face_img.shape)]  # workaround as the predict func expectsr multiple tensors and not a single one
            face_pred, face_scores = self.face_classifier.predict(torch.stack(face_imgs))
            face_scores = face_scores[0]

        embedding = torch.cat((reid_feat, face_scores))
        return embedding


def create_data_loader(reid_cfg, reid_model, device, dataloader_type, face_detector=None, face_classifier=None, dataset_path=None):
    if not dataset_path:
        dataset_path = reid_cfg.DATASETS.ROOT_DIR
    transforms_base = ReidTransforms(reid_cfg)
    reid_transform = transforms_base.build_transforms(is_train=False)
    num_workers = reid_cfg.DATALOADER.NUM_WORKERS
    if dataloader_type == 'ImageDataset':
        dataset = ImageDataset(dataset_path=dataset_path, reid_transform=reid_transform, reid_model=reid_model,
                               face_classifier=face_classifier, face_detector=face_detector, device=device)
    elif dataloader_type == 'ReIDClassifierDataset':
        dataset = ReIDClassifierDataset(dataset_path=dataset_path, reid_transform=reid_transform, reid_model=reid_model,
                                        device=device)
    else:
        raise Exception('Must provide a valid dataloader type. Options are: [ImageDataset, ReIDClassifierDataset]')
    data_loader = DataLoader(
        dataset,
        batch_size=reid_cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
    )
    return data_loader


def inference(model, data_loader, device):
    predictions = []
    model = model.cuda(device=device)

    for x in data_loader:
        model.eval()
        with torch.no_grad():
            pred = model(x)
            predictions.extend(pred)

    return predictions


class ScoresNet(nn.Module):
    def __init__(self):
        super(ScoresNet, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(1*(REID_EMBEDDING_SIZE+FACE_PREDICTION_SIZE), 512),
            nn.ReLU(),
            nn.Linear(512, FACE_PREDICTION_SIZE),
        )

    def forward(self, x):
        x = self.linear_layers(x)
        return x


def train(model, dataloader, num_epochs, criterion, optimizer, device):
    model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0
        # for x in tqdm.tqdm(iter(dataloader), total=len(dataloader)):
        for x in iter(dataloader):
            optimizer.zero_grad()
            data = x.get('features').to(device)
            labels = x.get('label').to(device)
            predictions = model(data)
            loss = criterion(predictions, labels.long())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {epoch_loss}')
        wandb.log({'classification_loss': epoch_loss})

    torch.save(model.state_dict(), CHECKPOINT_PATH)


def evaluate(model, dataloader, device):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    model.to(device)
    correct_predictions = 0
    for x in iter(dataloader):
        data = x.get('features').to(device)
        labels = x.get('label').to(device)
        predictions = model(data)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions += len(torch.where(predicted_labels == labels)[0])

    print(f'Total Accuracy: {correct_predictions/len(dataloader.dataset)}')


class ReIDClassifier(nn.Module):
    def __init__(self):
        super(ReIDClassifier, self).__init__()
        # self.classification_layer_1 = nn.Linear(REID_EMBEDDING_SIZE, 512)
        # self.activation = nn.ReLU()
        # self.classification_layer_2 = nn.Linear(512, OUTPUT_PREDICTION_SIZE)
        self.classification_layer = nn.Linear(REID_EMBEDDING_SIZE, OUTPUT_PREDICTION_SIZE)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.classification_layer_1(x)
        # x = self.activation(x)
        # x = self.classification_layer_2(x)
        x = self.classification_layer(x)
        x = self.softmax(x)
        return x
