import os
import sys

from FaceDetection.augmentions import smooth_sample_of_training, augment_training_set
from FaceDetection.data_handler import labelencode, load_data, FacesDataset, create_X_y_faces
from FaceDetection.evaluation import *

sys.path.append('DataProcessing')

from FaceDetection.faceDetector import FaceDetector


import pandas as pd
import copy
import pickle
import seaborn as sns
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from FaceDetection.facenet_pytorch import InceptionResnetV1, training
from torch import nn, optim
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class FaceClassifer():
    """
    The face Classifier is a CNN that utilizes InceptionResnetV1 architecture with a pretrain
    weights of vggface2 to fine-tune custom data
    """
    def __init__(self, num_classes, label_encoder, device='cuda:0', lr=0.00001):
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_ft = self._create_Incepction_Resnet_for_finetunning()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(),lr=lr)
        self.writer = SummaryWriter('/mnt/raid1/home/bar_cohen/FaceData/runs/images/')
        self.writer.iteration = 0
        self.writer.interval = 1
        self.softmax = nn.Softmax(dim=1)
        self.le = label_encoder
        self.exp_lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ft, [5,10])


    def _create_Incepction_Resnet_for_finetunning(self):
        model_ft = InceptionResnetV1(pretrained='vggface2', classify=True)
        model_ft.logits = torch.nn.Linear(512, self.num_classes)
        model_ft = model_ft.to(self.device)
        return model_ft

    def write_images_to_tensorboard(self,dl:DataLoader, grid_name:str,epoch:int):
        # run this on moba tensorboard --logdir="." --host 0.0.0.0
        images = [dl.dataset[np.random.randint(0, len(dl.dataset), 1)[0]][0] for i in range(dl.batch_size)]
        img_grid = make_grid(images, normalize=True)
        self.writer.add_image(grid_name, img_grid,global_step=epoch)

    def run_training(self,checkpoint_path, dl_train:DataLoader, dl_val:DataLoader,model_name:str, num_epochs=25) -> None:
        # dataloader_iter_train = iter(dl_train)
        # dataloader_iter_val = iter(dl_val)

        # images , target = next(dataloader_iter_val)
        # img_grid = make_grid(images, normalize=False)
        # self.writer.add_image('val_image_inputs', img_grid,global_step=0)
        #
        # img_grid = make_grid(images, normalize=True)
        # self.writer.add_image('val_image_inputs', img_grid,global_step=1)
        self.model_ft.eval()
        metrics = {'acc': training.accuracy}
        with torch.no_grad():
            eval_loss, eval_metric = training.pass_epoch(self.model_ft,
                                self.criterion,
                                dl_val, batch_metrics=metrics,
                                show_running=True,
                                device=self.device,
                                writer=self.writer)

        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            self.write_images_to_tensorboard(dl=dl_train,grid_name='train_sample', epoch=epoch)
            self.model_ft.train()
            training.pass_epoch(
                self.model_ft, self.criterion, dl_train, self.optimizer_ft,
                self.exp_lr_scheduler,
                batch_metrics=metrics, show_running=True, device=self.device,
                writer=self.writer
            )
            with torch.no_grad():
                self.write_images_to_tensorboard(dl=dl_val,grid_name='val_sample', epoch=epoch)
                self.model_ft.eval()
                eval_loss, eval_metric = training.pass_epoch(self.model_ft,
                                self.criterion,
                                dl_val, batch_metrics=metrics,
                                show_running=True,
                                device=self.device,
                                writer=self.writer)

            best_model_wts = copy.deepcopy(self.model_ft.state_dict())
            torch.save(best_model_wts, os.path.join(checkpoint_path ,f"{model_name}, {epoch + 1}.pth"))

    def create_data_loaders(self, X_train ,y_train, X_val ,y_val, X_test, y_test, data_path) -> (DataLoader, DataLoader, DataLoader):
        dl_train = DataLoader(FacesDataset(X_train, y_train), batch_size=100, shuffle=True)
        dl_val = DataLoader(FacesDataset(X_val, y_val), batch_size=100, shuffle=True)
        dl_test = DataLoader(FacesDataset(X_test, y_test), batch_size=100, shuffle=True)
        # must save datasets dls for reproducibility
        pickle.dump(dl_train, open(os.path.join(data_path, 'dl_train_1.pkl'), 'wb'))
        pickle.dump(dl_val, open(os.path.join(data_path, 'df_val_1.pkl'), 'wb'))
        pickle.dump(dl_test, open(os.path.join(data_path, 'df_test_1.pkl'), 'wb'))
        return dl_train, dl_val, dl_test

    def imshow(self, inp, labels=None):
        """Imshow for Tensor."""
        plt.figure()
        f , ax = plt.subplots(len(inp),1)
        for i, img in enumerate(inp):
            img = img.permute(1,2,0).int().numpy()
            ax[i].imshow(img)
            if labels:
                ax[i].set_ylabel(labels[i])
        plt.show()

    def predict(self, inputs):
        """Given input images as tensors, return predictions"""
        with torch.no_grad():
            outputs = self.softmax(self.model_ft(inputs.to(self.device)))
            preds = torch.argmax(outputs, dim=1)
            return preds, outputs

def main_train(data_path:str,run_name:str, reload_images_from_db:bool, recreate_data:bool,
               checkpoint_path:str, load_checkpoint:str, epochs=3, lr=0.001,
               save_images_path=''):
    """
    This is an example of a training pipeline. Insert a raw images path or existing
    faces data path to build a dataset and train a faceClassifer on it. Data will be saved
    as well metrics for the training.
    Returns:

    """
    fd = FaceDetector(faces_data_path=data_path, thresholds=[0.97,0.97,0.97],keep_all=False)
    print('loading finding faces base on DB tags, or reading from pickle...')
    high_conf_face_images = fd.filter_out_non_face_corps(recreate_data=reload_images_from_db, save_images_path=save_images_path) # keeps only face-present images in data
    num_classes = 21 # num of unique classes

    if recreate_data:
        print('Creating X,y datasets based on day splits')
        X_train ,y_train, X_val ,y_val, X_test, y_test = create_X_y_faces(high_conf_face_images,
                                                                          save_images_path='') # create an X,y dataset from filtered images

        print(f"Creates a label encoder and removes entered classes {num_classes} from dataset")
        X_train,y_train,X_val,y_val,X_test,y_test, le = labelencode(data_path, X_train, y_train, X_val, y_val, X_test, y_test, [])
        X_train, y_train = smooth_sample_of_training(X_train,y_train,max_sample_tresh=250)
        X_train, y_train = augment_training_set(X_train,y_train)
        fc = FaceClassifer(num_classes, le, device='cuda:1')
        print("Creates data loaders")
        dl_train,dl_val,dl_test = fc.create_data_loaders(X_train ,y_train, X_val ,y_val, X_test, y_test, data_path)

    else:
        print('Loading data given path ')
        le, dl_train, dl_val, dl_test = load_data(data_path)
        fc = FaceClassifer(num_classes, le, device='cuda:1', lr=lr)
        if load_checkpoint:
            print('checkpoint path received, loading..')
            fc.model_ft.load_state_dict(torch.load(os.path.join(checkpoint_path,load_checkpoint)))

    print(len(dl_train) * dl_train.batch_size, len(dl_val) * dl_val.batch_size, len(dl_test) * dl_test.batch_size)
    print('Begin Training')
    try:
        fc.run_training(checkpoint_path=checkpoint_path, dl_train=dl_train, dl_val=dl_val,model_name=run_name,num_epochs=epochs) # train the model
        print('Training Done')
    except KeyboardInterrupt:
        print('Training was interrupted by user')

    eval_faceClassifier(os.path.join(checkpoint_path, f"{run_name}, {epochs}.pth"))

if __name__ == '__main__':
    data_path = '/mnt/raid1/home/bar_cohen/FaceData'
    fd = FaceDetector(faces_data_path=data_path, thresholds=[0.97,0.97,0.97])
    # checkpoint = ""
    checkpoint_path = os.path.join("/mnt/raid1/home/bar_cohen/FaceData/checkpoints/")
    # main_train(data_path='/mnt/raid1/home/bar_cohen/FaceData/',
    #            run_name='min_face_20_lr_0.0001_skip_10frame',
    #            reload_images_from_db=False,
    #            recreate_data=True,
    #            checkpoint_path=checkpoint_path,
    #            load_checkpoint='',
    #            epochs=10,
    #            lr=0.00001,
    #            save_images_path='/mnt/raid1/home/bar_cohen/FaceData/DataSplitSmall')
    eval_faceClassifier(os.path.join(checkpoint_path, "min_face_20_lr_0.0001_skip_10frame, 4.pth"))
