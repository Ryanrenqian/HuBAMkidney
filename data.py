import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
import albumentations as A
import torch.utils.data as D
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from utils import *
import glob
import json
datas = {
    'HuBMAP': '/root/share/hubmap-kidney-segmentation/'
}

cache_dir='/root/workspace/human_klidney/cache'

def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)




def get_transform(name='default',resize=512):
    if name=='default':
        transform = A.Compose([
        A.Resize(resize,resize),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness(),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                    saturation=0.1, hue=0.1, always_apply=False, p=0.3),
            ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.0),
        A.ShiftScaleRotate(),])
    elif name=='train1':
        transform = A.Compose([
            A.RandomCrop(resize,resize,True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(brightness=0.07, contrast=0.07,saturation=0.1, hue=0.1, always_apply=False, p=0.3),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.ChannelShuffle(p=0.6)
        ])

    elif name=='val' or name=='test':
        transform = A.Compose([
                A.Resize(resize,resize)]
                )
    else:
        return None
    return transform


class HubDataset(D.Dataset):

    def __init__(self, config,root_dir, threshold = 100):
        self.path = root_dir
        self.config = config
        self.overlap = config['overlap']
        self.window = config['window']
        self.transform = get_transform(config['transform'],config['resize'])
        self.identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
        self.phase = config['phase']
        self.cache_dir = config['cache_dir']        
        if self.phase=='train' or self.phase=='val':
            self.csv = pd.read_csv(self.path + '/train.csv',
                                index_col=[0])
            self.threshold = threshold
        else:
            self.csv=None
        self.x, self.y,self.masks,self.slices, self.files  = [], [], [], [], []
        self.shape = {}
        if self.cache_dir:
            if self.check_cache():
                self.load_cache()
            else:
                os.system(f'mkdir -p {self.cache_dir}')
                self.build_cache()
        else:
            self.build_slices()
        
        self.len = len(self.slices)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])
        
    
    def build_slices(self):
        tag = self.phase
        if tag=='val':
            tag = 'train'
        for i, filepath in enumerate(glob.glob(self.path+f'{tag}/*.tiff')):
            filename = os.path.basename(filepath).split('.')[0]
            print(filepath,filename)
            with rasterio.open(filepath, transform =self.identity)  as dataset:
                flag =(self.phase=='train')
                if flag:
                    self.masks.append(rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape))
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)
                self.shape[filename] = dataset.shape
                for (x1,x2,y1,y2) in tqdm(slices):
                    if flag:
                        if not (self.masks[-1][x1:x2,y1:y2].sum() > self.threshold or np.random.randint(100) > 120):
                            continue
                        mask = self.masks[-1][x1:x2,y1:y2]
                    else:
                        mask = np.zeros((x2-x1,y2-y1))
                    image = dataset.read([1,2,3],window=Window.from_slices((x1,x2),(y1,y2)))
                    # if filter_roi(image):
                    #     continue
                    image = np.moveaxis(image, 0, -1)
                    self.x.append(image)
                    self.y.append(mask)
                    self.slices.append([i,filename,x1,x2,y1,y2])
                    


    
    def build_cache(self):
        self.build_slices()
        print('\nSave cache!')
        i = 0
        for _,filename,x1,x2,y1,y2 in tqdm(self.slices):
            # save images and mask in npz
            mask = self.y[i]
            image = self.x[i]
            npz_file = f'{self.cache_dir}/{filename}_{x1}_{x2}_{y1}_{y2}.npz'
            self.files.append(f'{filename}_{x1}_{x2}_{y1}_{y2}.npz')
            np.savez(npz_file,mask=mask,image=image)
            i+=1
            # save images and mask in png
            mask = Image.fromarray(mask).convert('RGB')
            save_mask = f'{self.cache_dir}/{filename}_{x1}_{x2}_{y1}_{y2}_mask.jpg'
            mask.save(save_mask)
            
            image = Image.fromarray(image).convert('RGB')
            save_image = f'{self.cache_dir}/{filename}_{x1}_{x2}_{y1}_{y2}_orgin.jpg'
            image.save(save_image)
            

        ## save cache data
        # self.slices = np.array(self.slices,dtype='int32,U3,int32,int32,int32,int32')
        np.savez(self.cache_dir+'/info.npz',shape=self.shape,files=self.files,slices=self.slices)
        

    def load_cache(self):
        data = np.load(f'{self.cache_dir}/info.npz',allow_pickle=True)
        self.slices = data['slices']
        self.files = data['files']
        self.shape = data['shape'][()]
    
    def check_cache(self):
        return  os.path.exists(f'{self.cache_dir}/info.npz')

    # get data operation
    def __getitem__(self, index):
        if self.check_cache():
            data = np.load(f'{self.cache_dir}/{self.files[index]}')
            image,mask = data['image'], data['mask']
        else:
            image, mask = self.x[index], self.y[index]
        if self.transform is not None:
            augments = self.transform(image=image, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None],index
        return self.as_tensor(image), self.as_tensor(mask),index
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def load_data(training_opt):
    ## Test data
    if training_opt['phase'] == 'test':
        orgin_dataset = HubDataset(config = training_opt ,root_dir = datas[training_opt['datasets']])
        return {'test':D.DataLoader(orgin_dataset,batch_size=training_opt['batch_size'], shuffle=False, num_workers=training_opt['num_workers'])}

    ## Train data
    orgin_dataset = HubDataset(config = training_opt ,root_dir = datas[training_opt['datasets']])
    indexes = [i for i in range(len(orgin_dataset))]
    # 
    if training_opt['phase'] == 'val':
        return {'test':D.DataLoader(orgin_dataset,  batch_size=training_opt['batch_size'], shuffle=False, num_workers=training_opt['num_workers'])}
    
    elif training_opt['strategy'] == 'randomn_split':
        '''randomnly select patches from each sample
        '''
        valid_idx,train_idx = [], []
        random.shuffle(indexes)
        endpoint = int(len(orgin_dataset) * 0.1)
        valid_idx,train_idx = indexes[:endpoint],indexes[endpoint:]
    elif training_opt['strategy'] == 'keep1wsi':
        valid_idx, train_idx = [], []
        for i in range(len(orgin_dataset)):
            if orgin_dataset.slices[i][1] == '1e2425f28':
                valid_idx.append(i)
            else:
                train_idx.append(i)
        print(f'sample 1e2425f28 for val')
    elif training_opt['strategy'] == 'keep2wsi':
        valid_idx, train_idx = [], []
        print(f'sample 0,7 for val')
        for i in range(len(orgin_dataset)):
            if orgin_dataset.slices[i][1] in ['2f6ecfcdf','1e2425f28']:
                valid_idx.append(i)
            else:
                train_idx.append(i)

    trainset = D.Subset(orgin_dataset,train_idx)
    validset = D.Subset(orgin_dataset,valid_idx)
    print(f"Size of trainset:\t{len(train_idx)}\n Size of validset:\t{len(valid_idx)}")
    # pdb.set_trace()
    train_loader = D.DataLoader(trainset, batch_size=training_opt['batch_size'], shuffle=True, num_workers=training_opt['num_workers'])
    valid_loader = D.DataLoader(validset, batch_size=training_opt['batch_size'], shuffle=False, num_workers=training_opt['num_workers'])
    return {'train': train_loader, 'val': valid_loader}