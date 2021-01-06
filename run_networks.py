import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
import cv2

class model():
    def __init__(self, config, data, test=False):
        self.config = config
        self.training_opt = self.config['training_opt']
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False
        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])

        # Initialize model
        self.init_models()
        
        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])
        self.init_optimizers(self.model_optim_params_dict)
        self.init_criterions()
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num  / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            
            
            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            print(f'Set log file:\t{self.log_file}')
            self.logger.log_cfg(self.config)
        else:
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            print(f'Set log file:\t{self.log_file}')
    
    def init_models(self,optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_dict = {}
        self.model_optim_named_params = {}

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():
            # Networks
            def_file = val['def_file']
            model_args = val['params']
            model_args.update({'test': self.test_mode})
            self.networks[key] = source_import(def_file).create_model(**model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                print('Freezing weights of module {}'.format(key))
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except final fc layer
                    if 'fc' not in param_name:
                        param.requires_grad = False
                print('=====> Freezing: {} | False'.format(key))
            
            if 'fix_flag' in val:
                for param_name, param in self.networks[key].named_parameters():
                    if val['fix_flag'] not in param_name:
                        param.requires_grad = False
                        print('=====> Freezing: {} | {}'.format(param_name, param.requires_grad))
                    else:
                        break
                
            if 'fix_set' in val:
                for fix_layer in val['fix_set']:
                    for param_name, param in self.networks[key].named_parameters():
                        if fix_layer == param_name:
                            param.requires_grad = False
                            print('=====> Freezing: {} | {}'.format(param_name, param.requires_grad))
                            continue


            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_named_params.update(dict(self.networks[key].named_parameters()))
            self.model_optim_params_dict[key] = {'params': self.networks[key].parameters()}
            self.model_optim_params_dict[key].update(optim_params)
            # self.model_optim_params_dict[key] = {'params': self.networks[key].parameters(),
            #                                     'lr': optim_params['lr'],
            #                                     'momentum': optim_params['momentum'],
            #                                     'weight_decay': optim_params['weight_decay']}
    
    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterion_weights = {}
        self.criterions = {}
        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())

            self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters()}.update(optim_params)]
                                # 'lr': optim_params['lr'],
                                # 'momentum': optim_params['momentum'],
                                # 'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params_dict):
        '''
        seperate backbone optimizer and classifier optimizer
        by Kaihua
        '''
        networks_defs = self.config['networks']
        self.model_optimizer_dict = {}
        self.model_scheduler_dict = {}
        # pdb.set_trace()
        for key, val in networks_defs.items():
            # optimizer
            if 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adam':
                print('=====> Using Adam optimizer')
                optimizer = optim.Adam([optim_params_dict[key],])
            elif 'optimizer' in self.training_opt and self.training_opt['optimizer'] == 'adamw':
                print('=====> Using AdamW optimizer')
                optimizer = optim.AdamW([optim_params_dict[key],])
            else:
                print('=====> Using SGD optimizer')
                optimizer = optim.SGD([optim_params_dict[key],])
            self.model_optimizer_dict[key] = optimizer
            # scheduler
            scheduler_params = val['scheduler_params']
            if scheduler_params['coslr']:
                print("===> Module {} : Using coslr eta_min={}".format(key, scheduler_params['endlr']))
                self.model_scheduler_dict[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                                    optimizer, self.training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
            elif scheduler_params['warmup']:
                print("===> Module {} : Using warmup".format(key))
                self.model_scheduler_dict[key] = WarmupMultiStepLR(optimizer, scheduler_params['lr_step'], 
                                                    gamma=scheduler_params['lr_factor'], warmup_epochs=scheduler_params['warm_epoch'])
            else:
                self.model_scheduler_dict[key] = optim.lr_scheduler.StepLR(optimizer,
                                                                                    step_size=scheduler_params['step_size'],
                                                                                    gamma=scheduler_params['gamma'])

        return
    
    def show_current_lr(self):
        max_lr = 0.0
        for key, val in self.model_optimizer_dict.items():
            lr_set = list(set([para['lr'] for para in val.param_groups]))
            if max(lr_set) > max_lr:
                max_lr = max(lr_set)
            lr_set = ','.join([str(i) for i in lr_set])
            print_str = ['=====> Current Learning Rate of model {} : {}'.format(key, str(lr_set))]
            print_write(print_str, self.log_file)
        return max_lr

    def batch_forward(self,inputs,labels, phase='train'):
        self.logits = self.networks['model'](inputs)['out']

    def batch_backward(self, print_grad=False):
        # Zero out optimizer gradients
        for key, optimizer in self.model_optimizer_dict.items():
            optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # display gradient
        if self.training_opt['display_grad']:
            print_grad_norm(self.model_optim_named_params, print_write, self.log_file, verbose=print_grad)
        # Step optimizers
        for key, optimizer in self.model_optimizer_dict.items():
            optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        self.loss = 0
        # pdb.set_trace()
        for criterion in self.criterions.keys():
            loss = self.criterions[criterion](self.logits, labels)
            loss *=  self.criterion_weights[criterion]
            self.loss += loss
    
    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y
    
    def train(self):
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)
        print_write(['Force shuffle in training??? --- ', self.do_shuffle], self.log_file)
        # Initialize best model
        best_model_weights = copy.deepcopy(self.networks['model'].state_dict())
        best_dice = 0.0
        best_loss = np.float('inf')
        best_epoch = 0
        # total_probs,total_labels = [],[]
        end_epoch = self.training_opt['num_epochs']
        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            for key, model in self.networks.items():
                # only train the module with lr > 0
                if self.config['networks'][key]['optim_params']['lr'] == 0.0:
                    print_write(['=====> module {} is set to eval due to 0.0 learning rate.'.format(key)], self.log_file)
                    model.eval()
                else:
                    model.train()

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            if epoch>1:
                for key, scheduler in self.model_scheduler_dict.items():
                    scheduler.step() 
                if self.criterion_optimizer:
                    self.criterion_optimizer_scheduler.step()
            # indicate current path
            print_write([self.training_opt['log_dir']], self.log_file)
            # print learning rate
            current_lr = self.show_current_lr()
            for step, (inputs, labels,_) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.float().cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels)
                    self.batch_loss(labels)
                    self.batch_backward(print_grad=(step % self.training_opt['display_grad_step'] == 0))
                    
                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:
                        minibatch_loss_total = self.loss.item()
                        self.probs = self.logits.sigmoid()
                        minibatch_dice = np_dice_score(torch2numpy(self.probs), torch2numpy(labels))

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_total: %.3f'
                                     % (minibatch_loss_total) if minibatch_loss_total else '',
                                     'Minibatch_Dice: %.3f'
                                     %(minibatch_dice)]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'DICE': minibatch_dice,
                        }
            # After every epoch, validation
            rsls = {'epoch': epoch}
            # rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            # rsls.update(rsls_train)
            rsls.update(rsls_eval)
            # Log results
            self.logger.log_acc(rsls)
            # Under validation, the best model need to be updated
            if self.eval_dice > best_dice:
                best_epoch_dice = epoch
                best_dice = self.eval_dice
            #     best_model_weights['model'] = copy.deepcopy(self.networks['model'].state_dict())
            if self.eval_loss<best_loss:
                best_loss = self.eval_loss
                best_epoch_loss = epoch
                best_model_weights['model'] = copy.deepcopy(self.networks['model'].state_dict())
            print('===> Saving checkpoint')
            self.save_latest(epoch)
        ## End training
        print('Training Complete.')

        print_str = ['Best validation Dice is %.3f at epoch %d' % (best_dice, best_epoch_dice)]
        print_str = ['Best validation Loss is %.3f at epoch %d' % (best_loss, best_epoch_loss)]
        print_write(print_str, self.log_file)
        # Save the best model
        self.save_model(best_epoch, best_model_weights, best_dice)

        # Test on the test set
        self.reset_model(best_model_weights)
        # self.eval('test' if 'test' in self.data else 'val')
        print('Done')



    def eval(self,phase='val',save=None,window=None):
        print_str = ['Phase Evaluation: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
        self.heatmap = None
        self.tp,self.tn = [],[]
        uions,overlaps =0, 0
        if phase == 'test':
            self.heatmap = {}
        loss = []
        # pdb.set_trace()
        # samples = []
        with torch.no_grad():
            for inputs, labels,indexes in tqdm(self.data[phase]):
                inputs, labels = inputs.cuda(), labels.float().cuda()
                # If on training phase, enable gradients
                self.batch_forward(inputs,labels)
                # pdb.set_trace()
                self.batch_loss(labels)
                self.probs = self.logits.sigmoid()

                self.batch_loss(labels)
                loss.append(self.loss.item())
                probs,labels = torch2numpy(self.probs),torch2numpy(labels)
                uion,overlap = return_cover(probs,labels)
                uions += uion
                overlaps += overlap
                tp,tn = np_accuracy(probs,labels)
                self.tp.append(tp)
                self.tn.append(tn)
                if self.heatmap!=None :
                    batch_size = inputs.size()[0]
                    for i in range(batch_size):
                        _,filename,x1,x2,y1,y2 = self.data[phase].dataset.slices[indexes[i]]
                        # init heatmap
                        if filename not in self.heatmap.keys():
                            shape = self.data[phase].dataset.shape[filename]
                            self.heatmap[filename] = np.zeros(shape) # H*W
                        prob = cv2.resize(probs[i][0], (window, window))
                        self.heatmap[filename][int(x1):int(x2),int(y1):int(y2)] = (prob>0.5).astype(np.uint8)
            
            if save:
                save_mask = f'{save}/'
                os.system(f'mkdir -p {save_mask}')
                subm={}
                for i,filename in enumerate(self.heatmap.keys()):
                    np.save(f'{save_mask}/{filename}.npy' ,self.heatmap[filename])
                    subm[i]={'id': filename, 'predicted': rle_encode(self.heatmap[filename])}
                submission = pd.DataFrame.from_dict(subm, orient='index')
                submission.to_csv(f'{save}/mask.csv', index=False)

        # Calculate and pringt the overall Dice
        self.eval_dice = 2*overlaps/(uions+0.0001)
        self.eval_loss = np.mean(np.array(loss))
        self.tp = np.mean(self.tp)
        self.tn = np.mean(self.tn)
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     f'accuracy TP: {self.tp:.3f}, FP: {self.tn:.3f}',
                     '\n\n',
                     f'Loss:\t {self.eval_loss}',
                     '\n\n',
                     'Evaluation_Dice: %.3f'
                     % (self.eval_dice),
                     '\n']
        print_write(print_str, self.log_file)
        rsl = {phase + '_dice': self.eval_dice,phase+'_loss':self.eval_loss}
        return rsl
    
    def reset_model(self,model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self,model_dir=None):
        print('validation on the best model')
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        print('Loading model from %s' % (model_dir))
        checkpoint = torch.load(model_dir)
        try:         
            model_state = checkpoint['state_dict']['model']
        except KeyError:
            model_state = checkpoint['state_dict']
        self.networks['model'].load_state_dict(model_state)

    def save_latest(self,epoch):
        model_weights = {}
        model_weights['model'] = copy.deepcopy(self.networks['model'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights,
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    def save_model(self, best_epoch, best_model_weights, best_dice):
        model_states = {
                'epoch': best_epoch,
                'state_dict': best_model_weights,
                'dice': best_dice,
                }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)




