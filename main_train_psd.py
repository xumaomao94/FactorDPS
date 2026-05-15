import argparse
import core.logger as Logger
import torch
import logging
from core.wandb_logger import WandbLogger
from core.datasets import PSDDataset
from torch.utils.data import DataLoader
import diffusionmodel as Model
from utils.visualize import get_psd_as_grid
import wandb
import matplotlib.pyplot as plt
import os
import numpy as np

def log_psd_as_grid(psd_data, caption, title=None):
    # grid = make_grid(images, nrow=5, normalize=False)
    # wandb.log({"images": [wandb.Image(grid, caption=caption)]})
    fig = get_psd_as_grid(psd_data, data_range=[None, None], title=title)
    wandb.log({caption: [wandb.Image(fig, caption=caption)]})
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/PSD_DDPM.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Setup logger
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger(
        'sampling', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None

    '''Data sets and loaders'''
    train_dataset = PSDDataset(num_peaks_per_psd=opt['datasets']['num_peaks'],
                               K=opt['datasets']['K'],
                               basis_type=opt['datasets']['basis_type'],
                               num_samples=opt['datasets']['train_data_num'],
                               save_dir = opt['path']['dataprocessor'])
    
    train_loader = DataLoader(train_dataset, batch_size=opt['train']['batch_size'],
                              num_workers=4, shuffle=True)
    logger.info('Training dataset initialized.')


    # model
    diffusion = Model.create_PSD_model(opt)
    logger.info('Initial Model Finished')




     # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log                
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                
                 # validation
                if current_step % 2500 == 1:
                    if wandb_logger:
                        train_data = train_data["PSD"].cpu()
                        train_data = train_data[:8] if train_data.size(0) > 8 else train_data
                        log_psd_as_grid(train_dataset.reverse_transform(train_data), "Training Data")

                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    

                    mat_result_path = '{}/{}'.format(opt['path']
                                                 ['mat_results'], current_epoch)
                    os.makedirs(mat_result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    val_data = []
                    for idx in range(8):
                        
                        diffusion.sample(continous=False)
                        visuals = diffusion.get_current_visuals(sample=True)
                        
                        val_data.append(visuals['SAMPLE'].cpu().numpy())


                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    
                    val_data = np.concatenate(val_data, axis=0)
                    if wandb_logger:
                        log_psd_as_grid(train_dataset.reverse_transform(val_data), "Sampled Data")

                if current_step % 25000 == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

        # save model
        logger.info('End of training.')