import os
import sys
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
    def __init__(self, num_classes, label_encoder, device='cuda:0', lr=0.00001, exp_name=''):
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_ft = self._create_Incepction_Resnet_for_finetunning()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(),lr=lr)
        self.writer = SummaryWriter(f'/mnt/raid1/home/bar_cohen/FaceData/runs/images/{exp_name}')
        self.writer.iteration = 0
        self.writer.interval = 1
        self.softmax = nn.Softmax(dim=1)
        self.le = label_encoder
        self.exp_lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ft, [5,10])

    def _create_Incepction_Resnet_for_finetunning(self):
        model_ft = InceptionResnetV1(pretrained='vggface2', classify=True,num_classes=self.num_classes, dropout_prob=0.6)
        # model_ft.logits = torch.nn.Linear(512, self.num_classes)
        model_ft = model_ft.to(self.device)
        return model_ft

    def write_images_to_tensorboard(self,dl:DataLoader, grid_name:str,epoch:int):
        # run this on moba tensorboard --logdir="." --host 0.0.0.0
        images = [dl.dataset[np.random.randint(0, len(dl.dataset), 1)[0]][0] for i in range(dl.batch_size)]
        img_grid = make_grid(images, normalize=True)
        self.writer.add_image(grid_name, img_grid,global_step=epoch)

    def run_training(self,checkpoint_path, dl_train:DataLoader, dl_val:DataLoader,dl_test:DataLoader,
                     model_name:str, num_epochs=25) -> None:
        # self.write_images_to_tensorboard(dl=dl_train, grid_name='train_sample', epoch=0)
        metrics = {'acc': training.accuracy}
        self.model_ft.eval()
        with torch.no_grad():
            eval_loss, eval_metric = training.pass_epoch(self.model_ft,
                                self.criterion,
                                dl_val, batch_metrics=metrics,
                                show_running=True,
                                device=self.device,
                                writer=self.writer)


        for epoch in range(0,num_epochs):
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
            torch.save(best_model_wts, os.path.join(checkpoint_path ,f"{model_name}, {epoch}.pth"))


        # with torch.no_grad():
        #     self.write_images_to_tensorboard(dl=dl_test,grid_name='test_sample', epoch=num_epochs)
        #     self.model_ft.eval()
        #     eval_loss, eval_metric = training.pass_epoch(self.model_ft,
        #                     self.criterion,
        #                     dl_test, batch_metrics=metrics,
        #                     show_running=True,
        #                     device=self.device,
        #                    )
        #     self.writer.add_scalars('test_acc', eval_metric, num_epochs)



    def create_data_loaders(self, X_train ,y_train, X_val ,y_val, X_test, y_test, data_path) -> (DataLoader, DataLoader, DataLoader):
        from FaceDetection.data_handler import FacesDataset
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
        self.model_ft.eval()
        with torch.no_grad():
            outputs = self.softmax(self.model_ft(inputs.to(self.device)))
            preds = torch.argmax(outputs, dim=1)
            return preds, outputs

def main_train(data_path:str,run_name:str, reload_images_from_db:bool, recreate_data:bool,augment_data:bool,
               checkpoint_path:str, load_checkpoint:str, epochs=3, lr=0.001,
               save_images_path='', max_sample_threshold=0, device='cuda:1'):
    from FaceDetection.augmentions import smooth_sample_of_training, augment_training_set
    from FaceDetection.data_handler import labelencode,load_data, create_X_y_faces, load_old_data
    from FaceDetection.evaluation import build_samples_hist, per_id_test_acc
    from FaceDetection.faceDetector import FaceDetector
    """
    This is an example of a training pipeline. Insert a raw images path or existing
    faces data path to build a dataset and train a faceClassifer on it. Data will be saved
    as well metrics for the training.
    Returns:

    """
    fd = FaceDetector(faces_data_path=data_path, thresholds=[0.97,0.97,0.97],keep_all=False, device=device)
    print('loading finding faces base on DB tags, or reading from pickle...')
    high_conf_face_images = fd.filter_out_non_face_corps(recreate_data=reload_images_from_db, save_images_path=save_images_path) # keeps only face-present images in data
    num_classes = 21 # num of unique classes

    if recreate_data:
        print('Creating X,y datasets based on day splits')
        X_train ,y_train, X_val ,y_val, X_test, y_test = create_X_y_faces(high_conf_face_images,
                                                                          save_images_path='') # create an X,y dataset from filtered images
        print('Adding Train Data from previously labeled data-set')
        x_train_add , y_train_add = load_old_data('/mnt/raid1/home/bar_cohen/FaceData/old_data_for_train/', reload_images=reload_images_from_db)
        X_train.extend(x_train_add)
        y_train.extend(y_train_add)
        print(f"Creates a label encoder and removes entered classes {num_classes} from dataset")
        X_train,y_train,X_val,y_val,X_test,y_test, le = labelencode(data_path, X_train, y_train, X_val, y_val, X_test, y_test, [])
        build_samples_hist(le, y_train, run_name, 'Train')
        build_samples_hist(le, y_val, run_name, 'Val')
        build_samples_hist(le, y_test, run_name, 'Test')

        if max_sample_threshold > 0:
            print(f'Smoothing according to max_sample_threshold {max_sample_threshold}')
            X_train, y_train = smooth_sample_of_training(X_train,y_train,max_sample_threshold=max_sample_threshold) # before augs
        print('Augmenting data set..')
        X_train, y_train = augment_training_set(X_train,y_train, augment_data=augment_data)
        fc = FaceClassifer(num_classes, le, device=device,exp_name=run_name)
        print("Creates data loaders")
        dl_train,dl_val,dl_test = fc.create_data_loaders(X_train ,y_train, X_val ,y_val, X_test, y_test, data_path)

    else:
        print('Loading data given path ')
        le, dl_train, dl_val, dl_test = load_data(data_path)
        fc = FaceClassifer(num_classes, le, device=device, lr=lr, exp_name=run_name)
        if load_checkpoint:
            print('checkpoint path received, loading..')
            fc.model_ft.load_state_dict(torch.load(os.path.join(checkpoint_path,load_checkpoint)))

    print(len(dl_train) * dl_train.batch_size, len(dl_val) * dl_val.batch_size, len(dl_test) * dl_test.batch_size)
    print('Begin Training')
    try:
        fc.run_training(checkpoint_path=checkpoint_path, dl_train=dl_train, dl_val=dl_val,dl_test=dl_test,
                        model_name=run_name,num_epochs=epochs) # train the model
        print('Training Done')
    except KeyboardInterrupt:
        print('Training was interrupted by user')
        return

    per_id_test_acc(fc, le, run_name, dl_train, 'Train')
    per_id_test_acc(fc, le, run_name, dl_val, 'Val')
    per_id_test_acc(fc, le, run_name, dl_test, 'Test')
    # eval_faceClassifier(run_name,os.path.join(checkpoint_path, f"{run_name}, {epochs-1}.pth"))

if __name__ == '__main__':
    pass
    from FaceDetection.evaluation import eval_faceClassifier
    # from FaceDetection.faceDetector import FaceDetector
    # data_path = '/mnt/raid1/home/bar_cohen/FaceData'
    # fd = FaceDetector(faces_data_path=data_path, thresholds=[0.97,0.97,0.97], device='cuda:0')
    # # checkpoint = ""
    # checkpoint_path = os.path.join("/mnt/raid1/home/bar_cohen/FaceData/checkpoints/")
    # for with_augs in [True,False]:
    #     for max_sample_threshold in [0,500,400,300]:
    #         for lr in [0.001,0.0001,0.00001]:
    #             run_name = f"FULL_DATA_augs:{with_augs}_lr:{lr}_{max_sample_threshold}"
    #             print(run_name)
    #             main_train(data_path='/mnt/raid1/home/bar_cohen/FaceData/',
    #                        run_name=run_name,
    #                        reload_images_from_db=False,
    #                        recreate_data=True,
    #                        augment_data = with_augs,
    #                        checkpoint_path=checkpoint_path,
    #                        load_checkpoint='',
    #                        epochs=5,
    #                        lr=lr,
    #                        save_images_path='',
    #                        max_sample_threshold=max_sample_threshold,
    #                        device='cuda:0')
    # eval_faceClassifier(exp_name='test',
    #                     checkpoint_path=os.path.join(checkpoint_path, "/mnt/raid1/home/bar_cohen/FaceData/checkpoints/all_data_with_augs_sample_threshold_500_weight_decay_dropout_0.8, 3.pth"))
#