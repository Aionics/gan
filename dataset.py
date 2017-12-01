import torch.utils.data as tdata
from PIL import Image, ImageDraw, ImageFont
import os
from scipy.misc import imread
from scipy.misc import imresize

import numpy as np

def pathToNumpy(file):
    img = Image.open(file)

    width, height = img.size
    left = (width - 256)/2
    top = (height - 256)/2
    right = (width + 256)/2
    bottom = (height + 256)/2


    imagenp = np.asarray( img.crop((left, top, right, bottom)) )
    return np.transpose(imagenp, (2,0,1)) # HxWxC ---> CxHxW

def loadWholeDatasetToMemory(input_path, filenames):
    images = []
    for i in range(len(filenames)):
        img_fn = os.path.join(input_path, filenames[i])
        npimage = pathToNumpy(img_fn)
        images.append(npimage)
    return np.array(images)

class datasetLoader(tdata.Dataset):
    def __init__(self, image_dir, subfolder='train', preload=False):
        super(datasetLoader, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.preload = preload
        self.preloaded_dataset = None if not preload else loadWholeDatasetToMemory(self.input_path, self.image_filenames)

    def __getitem__(self, index):
        if not self.preload:

            real = imread(os.path.join(self.input_path, self.image_filenames[index]))
            real = imresize(real, (256, 256), interp='nearest', mode=None)
            # img_fn = os.path.join(self.input_path, self.image_filenames[index])
            # img = Image.open(img_fn)
            #
            # width, height = img.size
            # left = (width - 256)/2
            # top = (height - 256)/2
            # right = (width + 256)/2
            # bottom = (height + 256)/2
            #
            #
            # real = np.asarray( img.crop((left, top, right, bottom)) )
            # print(real.shape)
            real = np.transpose(real, (2,0,1)) # HxWxC ---> CxHxW
            noise = np.random.uniform(low=0.0, high=1.0, size=(100, 1, 1))

            return (real, noise)

        else:
            real = self.preloaded_dataset[index]
            noise = np.random.uniform(low=0.0, high=1.0, size=(100, 1, 1))

            return (real, noise)


    def __len__(self):
        return len(self.image_filenames)
