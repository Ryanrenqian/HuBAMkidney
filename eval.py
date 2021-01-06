from utils import *
from glob import glob
from data import datas
import rasterio
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, type=str)
args = parser.parse_args()
datasize = {}
csv = pd.read_csv(f'{datas["HuBMAP"]}/train.csv',index_col=[0])
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
print(f'Loading Dataset: {datas["HuBMAP"]}\n\n')
for filepath in glob(f'{datas["HuBMAP"]}/train/*.tiff'):
    filename = os.path.basename(filepath).split('.')[0]
    print(filename)
    with rasterio.open(filepath, transform =identity)  as dataset:
        datasize[filename] = dataset.shape
log_file = args.dir+'/result_eval.txt'

def eval_dice(result_folder, log_file, phase='mask'):
    uions,overlaps=0,0
    print(f'\n\nLoading Result Folder,{result_folder}')
    # print(f'{result_folder}/*.npy')
    for filepath in glob(f'{result_folder}/*.npy'):
        print(filepath)
        filename = os.path.basename(filepath).split('.')[0]
        pred = np.load(filepath)
        mask = rle_decode(csv.loc[filename, 'encoding'], datasize[filename])
        uion,overlap =return_cover(pred,mask)
        dice =2*overlap/uion
        print_write([f'Dice {filename}:\t{dice:.4f}'], log_file)
        uions+=uion
        overlaps+=overlap
    dice = 2 * overlaps / uions
    print_write([f'Dice in sum:\t{dice:.4f}'], log_file)

eval_dice(args.dir,log_file)