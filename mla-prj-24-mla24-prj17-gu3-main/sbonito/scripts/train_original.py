"""Script used for training for the CNN analysis experiment
"""

import os
import sys
import shutil

from torch.optim import lr_scheduler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../bonito')))


from classes import BaseNanoporeDataset
from schedulers import GradualWarmupScheduler
from constants import NON_RECURRENT_DECODING_DICT, NON_RECURRENT_ENCODING_DICT, RECURRENT_DECODING_DICT, RECURRENT_ENCODING_DICT
from utils import print_architecture

import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import pandas as pd
import time

params_l2mu_leaky = {
      "hidden_size": 384,
      "memory_size": 20,
      "order": 3,#più alto il polinomio per approssimare = 5/10
      "theta": 1,#lunghezza nel tempo campioni
      "beta_spk_u": 0.6,
      "threshold_spk_u": 0.15,
      "beta_spk_h": 0.3,
      "threshold_spk_h": 0.35,
      "beta_spk_m": 0.75,
      "threshold_spk_m": 0.6,
      "beta_spk_output": 0.7,
      "threshold_spk_output": 0.35
  }




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_log_df(losses_keys, metrics_keys):
    """Creates a data.frame to store the logging values
    """
    
    header = ['epoch', # epoch number
              'step',  # step number
              'time']  # time it took
    # add losses and metrics for train and validation
    for k in losses_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    for k in metrics_keys:
        header.append(k + '.train')
        header.append(k + '.val')
    # whether a checkpoint was saved at this step
    header.append('lr')
    header.append('checkpoint')
    
    log_dict = dict()
    for k in header:
        log_dict[k] = [None]
    return pd.DataFrame(log_dict)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help='Path where the data for the dataloaders is stored')
    parser.add_argument("--output-dir", type=str, help='Path where the model is saved')
    parser.add_argument("--model", type=str, choices=[
        'bonito',
        'catcaller',
        'causalcall',
        'mincall',
        'sacall',
        'urnano',
        'halcyon',
        'bonitosnn',
        'bonitospikeconv',
        'bonitospikelin',
        'S5'
    ], help='Model')
    parser.add_argument("--window-size", type=int, choices=[400, 1000, 2000, 4000], help='Window size for the data')
    parser.add_argument("--num-epochs", type=int, default = 5)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--starting-lr", type=float, default = 0.001)
    parser.add_argument("--warmup-steps", type=int, default = 5000)
    parser.add_argument("--use-scaler", action='store_true', help='use 16bit float precision')
    parser.add_argument("--overwrite", action='store_true', help='delete existing files in folder')
    parser.add_argument("--checkpoint", type=str, help='checkpoint file to resume training')
    parser.add_argument("--nlstm",type=int,default=0,choices=[0,1,2,3,4],help='number of lstm blocks must be between 0 and 4')
    parser.add_argument("--conv-th",type=float,default=0.05,help='Neuron threshold for spikeconv and spikelin')
    parser.add_argument("--slstm-th",type=float,default=0.05,help='slstm threshold')
    parser.add_argument("--nl2mu", type=int, default=0,help = "number of l2mu istances")
    parser.add_argument("--one-conv", type= bool, default= False,help = "set to true for a 1 convolution model")
    args = parser.parse_args()
    
    validate_every = 100
    checkpoint_every = 20000

    data_dir = args.data_dir

    """
    if args.model == 'halcyon':
        from halcyon.model import HalcyonModelS2S as Model 
        decoding_dict = RECURRENT_DECODING_DICT
        encoding_dict = RECURRENT_ENCODING_DICT
        s2s = True
    else:
    """
    decoding_dict = NON_RECURRENT_DECODING_DICT
    encoding_dict = NON_RECURRENT_ENCODING_DICT
    s2s = False
    if args.model == 'bonito':
        from bonito.model import BonitoModel as Model
    elif args.model == 's5':
        from bonito.model import S5Model as Model
    elif args.model == 'bonitosnn':
        from bonitosnn.model.snn_model import BonitoSNNModel as Model 
    elif args.model == 'bonitospikeconv':
        from bonitosnn.model.snn_model import BonitoSpikeConv as Model
    elif args.model == 'bonitospikelin':
        from bonitosnn.model.snn_model import BonitoSpikeLin as Model
    l2mu_dict = {"number" : args.nl2mu,
                "type" : "leaky",
                "params" : params_l2mu_leaky}
    print('Creating dataset')
    dataset = BaseNanoporeDataset(
        data_dir = data_dir, 
        decoding_dict = decoding_dict, 
        encoding_dict = encoding_dict, 
        split = 0.95, 
        shuffle = True, 
        seed = 1,
        s2s = s2s,
    )

    dataloader_train = DataLoader(
        dataset, 
        batch_size = args.batch_size, 
        sampler = dataset.train_sampler, 
        num_workers = 2 #TODO: change to 4 in original (2 only for Colab)
    )
    dataloader_validation = DataLoader(
        dataset, 
        batch_size = args.batch_size, 
        sampler = dataset.validation_sampler, 
        num_workers = 2  #TODO: change to 4 in original (2 only for Colab)
    )

    
    if args.use_scaler:
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        use_amp = False
        scaler = None

    print('Creating model: ' + args.model)
    model = Model(
              load_default = True,
              device = device,
              dataloader_train = dataloader_train, 
              dataloader_validation = dataloader_validation,
              scaler = None,
              use_amp = False,
              nlstm=0,
            #   l2mu = l2mu_dict,     #TODO: I have doubts that removing l2mu can be an intelligent solution, even if it is not actually used!
            #   nconv = args.one_conv  #TODO: I have doubts that removing one_conv can be an intelligent solution, even if it is not actually used!
        )
    '''model = Model(
          load_default = True,
          device = device,
          dataloader_train = dataloader_train, 
          dataloader_validation = dataloader_validation, 
          scaler = scaler,
          use_amp = use_amp,
          nlstm=args.nlstm,
          slstm_threshold=args.slstm_th,
          conv_threshold=args.conv_th
      )'''
    model = model.to(device)
  
    print_architecture(model)

    print('Creating optimization')
    ##    OPTIMIZATION     #############################################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.starting_lr)
    total_steps =  (len(dataset.train_idxs)*args.num_epochs)/args.batch_size
    cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_steps, eta_min=0.00001, last_epoch=-1, verbose=False)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier = 1.0, total_epoch = args.warmup_steps, after_scheduler=cosine_lr)
    schedulers = {'lr_scheduler': lr_scheduler}
    clipping_value = 2
    use_sam = False


    ##   MODEL PART2        #############################################
    model.optimizer = optimizer
    model.schedulers = schedulers
    model.clipping_value = clipping_value
    model.use_sam = use_sam

    if args.checkpoint is not None:
        model.load(args.checkpoint, initialize_lazy = True)
        model.to(device)

    print('Creating outputs')
    # output stuff
    output_dir = args.output_dir
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')

    # check output dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(checkpoints_dir)
    else:
        if args.overwrite:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
            os.mkdir(checkpoints_dir)
        else:
            if len(os.listdir(output_dir)) > 0:
                raise FileExistsError('Output dir contains files')
            else:
                os.mkdir(checkpoints_dir)
    
    # keep track of losses and metrics to take the average
    train_results = dict()
    
    print('Training')
    total_num_steps = 1
    for epoch_num in range(args.num_epochs):
        print(f"epoch {epoch_num}")
        loader_train = model.dataloader_train
        loader_validation = model.dataloader_validation
        # use this to restart the in case we finish all the validation data
        validation_iterator = iter(loader_validation) 
        
        st_time = time.time()
        # iterate over the train data
        for train_batch_num, train_batch in enumerate(loader_train):
            #print(f" batch {train_batch_num}/{len(loader_train) - 1}")
            losses, predictions = model.train_step(train_batch)
            total_num_steps += 1
            
            for k, v in losses.items():
                if k not in train_results.keys():
                    train_results[k] = list()
                train_results[k].append(v)
            
            if total_num_steps % validate_every == 0:
                
                # calculate accuracy for the training only here since doing for every batch
                # is expensive and slow...
                predictions_decoded = model.decode(predictions, greedy = True)
                metrics = model.evaluate(train_batch, predictions_decoded)
                
                # log the train results
                log_df = generate_log_df(list(losses.keys()), list(metrics.keys()))
                for k, v in train_results.items():
                    log_df[k + '.train'] = np.mean(v)
                for k, v in metrics.items():
                    log_df[k + '.train'] = np.mean(v)
                train_results = dict() # reset the dict
                
                try:
                    validation_batch = next(validation_iterator)
                except StopIteration:
                    validation_iterator = iter(loader_validation)
                    validation_batch = next(validation_iterator)
                                
                # calculate and log the validation results
                losses, predictions = model.validation_step(validation_batch)
                predictions_decoded = model.decode(predictions, greedy = True)
                metrics = model.evaluate(validation_batch, predictions_decoded)
                
                for k, v in losses.items():
                    log_df[k + '.val'] = v # do not need the mean as we only did it once
                for k, v in metrics.items():
                    log_df[k + '.val'] = np.mean(v)
                    
                # calculate time it took since last validation step
                log_df['epoch'] = str(epoch_num)
                log_df['step'] = str(total_num_steps)
                log_df['time'] = int(time.time() - st_time)
                for param_group in model.optimizer.param_groups:
                    log_df['lr'] = param_group['lr']
                st_time = time.time()
                    
                # save the model if we are at a saving step
                if total_num_steps % checkpoint_every == 0:
                    log_df['checkpoint'] = 'yes'
                    model.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(total_num_steps) + '.pt'))
                else:
                    log_df['checkpoint'] = 'no'
                
                # write to log
                if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                    log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                  header=True, index=False)
                else: # else it exists so append without writing the header
                    log_df.to_csv(os.path.join(output_dir, 'train.log'), 
                                  mode='a', header=False, index=False)
                    
                # write results to console
                print(log_df)
                
    
    model.save(os.path.join(checkpoints_dir, 'checkpoint_' + str(total_num_steps) + '.pt'))