import os
from PIL import Image

class ImagesDataset(object):
    def __init__(self, config,is_train = False, transforms=None):
        self.config = config
        self.transforms = transforms
        ### all files in the root
        self.imgs = []
        for class_i in self.config['data']['class_names']:
            if is_train:
                folder_img_path = os.path.join(os.path.abspath(self.config['data']['data_dir']),class_i,'train')
            else:
                folder_img_path = os.path.join(os.path.abspath(self.config['data']['data_dir']),class_i,'test')
            ### corresponding paths to images
            class_imgs = [os.path.join(folder_img_path,img_i) for img_i in os.listdir(folder_img_path) if not img_i.startswith('.')]
            self.imgs.extend(class_imgs)
    def __getitem__(self, idx):
        ### load images and masks
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        ### get the parent directory of the file
        label_dir = os.path.split(os.path.dirname(os.path.dirname(img_path)))[-1]
        label = self.config['data']['class_names'].index(label_dir)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)