from torchvision import transforms as T
import PIL
import numpy as np

def normalize_image(img):
    transform = T.Compose([
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(img)

def crop_top_third_and_sides(img, is_PIL_input):
    # this assumes img is a PIL obj
    if not is_PIL_input:
        img = PIL.Image.fromarray(img)
    width, height = img.size
    cropped_img = img.crop((width*0.2, 0, width*0.8, height*0.3))
    return cropped_img

def detect_single_face_inv_norm(self, img):
    # use face detector to find a single face in an image, rests to entered init of keeping face after change
    self.facenet_detecor.keep_all = False
    comp = T.Compose([
        T.ToTensor(),
        lambda x:x*255,
    ])
    new_im = comp(img).permute(1,2,0).int()
    ret, prob = self.facenet_detecor(new_im, return_prob=True)
    self.facenet_detecor.keep_all = self.keep_all
    return ret , prob


def smooth_sample_of_training(x_train, y_train, max_sample_threshold):
    inds_to_take = []
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    for k in set(y_train):
        inds_of_label_k = np.where(y_train == k)[0]
        if len(inds_of_label_k) > max_sample_threshold:
            # k_label_images, k_label_tags = x_train[inds_of_label_k], y_train[inds_of_label_k]
            sub_inds =list(np.random.choice(inds_of_label_k, size=max_sample_threshold))
            inds_to_take.extend(sub_inds)
        else:
            inds_to_take.extend(list(inds_of_label_k))
    return x_train[inds_to_take], y_train[inds_to_take]


def augment_training_set(x_train:[], y_train:[]):
    augmenters = [T.RandomHorizontalFlip(),
                 T.ColorJitter(),
                 T.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5)),
                 T.Grayscale(num_output_channels=3),
                 T.RandomErasing(),
                 T.RandomRotation(degrees=(0,45))
                 ]
    augmented_x = []
    augmented_y = []
    for img , label in zip(x_train, y_train):
        augmented_x.append(normalize_image(img))
        augmented_y.append(label)
        for aug in augmenters:
            augmented_x.append(normalize_image(aug(img)))
            augmented_y.append(label)
    return augmented_x, augmented_y


