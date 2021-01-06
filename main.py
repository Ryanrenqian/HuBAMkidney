import torch
import os
import yaml
import pprint
import argparse
from utils import source_import,update
import warnings
from data import load_data
from run_networks import model
import numpy as np
data_root = {'humap': '/root/share/hubmap-kidney-segmentation/'}


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--val', default=False, action='store_true')
parser.add_argument('--model_dir', type=str, default=None)
args = parser.parse_args()
# ============================================================================
# Random Seed
import random
if args.seed:
    print('=======> Using Fixed Random Seed <========')
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f)
config = update(config, args)

test_mode = args.test
val_mode = args.val


if not test_mode and not val_mode:
    opt = config['training_opt']
    if not os.path.isdir(opt['log_dir']):
        os.makedirs(opt['log_dir'])
    pprint.pprint(config)
    data = load_data(opt)
    training_model = model(config,data,test=False)
    training_model.train()
if val_mode:
    opt = config['val_opt']
    data = load_data(opt)
    training_model = model(config,data,test=True)
    # training_model.load_model(args.model_dir)
    # pdb.set_trace()
    training_model.eval('test',save=opt['log_dir'],window=opt['window'])
if test_mode:
    opt = config['testting_opt']
    data = load_data(opt)
    training_model = model(config,data,test=True)
    # load checkpoints
    # training_model.load_model(args.model_dir)
    training_model.eval('test',save=opt['log_dir'],window=opt['window'])
print('All COMPLETED!!!')