import os
import numpy as np
import diffusionmodel as Model
import argparse
import logging

import torch
from torch.utils.data import DataLoader
import core.HSImetrics as Metrics

import core.logger as Logger
import core.vis as visual
from core.wandb_logger import WandbLogger
from core.datasets import SLFDataset, RadioMapSeerSLFDataset, RadioMapSeerRMDataset
import wandb
from torchvision.utils import make_grid
from utils.visualize import get_slf_as_grid
import matplotlib.pyplot as plt


def log_images_as_grid(images, caption, title=None, plot_type='SLF', data_range=[None, None]):
    # grid = make_grid(images, nrow=5, normalize=False)
    # wandb.log({"images": [wandb.Image(grid, caption=caption)]})
    fig = get_slf_as_grid(images, data_range=data_range,
                          title=title, plot_type=plot_type)
    wandb.log({caption: [wandb.Image(fig, caption=caption)]})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SLF_RadioMapSeer_DDPM_conditional_625Buildings.json',
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

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger(
        'sampling', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    # Log process ID and GPU status for debugging
    import os
    current_pid = os.getpid()
    logger.info(f"Process ID: {current_pid}")
    logger.info(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i} - Total: {total:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {total-reserved:.2f}GB")

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        val_step = 0
    else:
        wandb_logger = None

    train_path = opt['datasets']['data_root_dir']
    if opt["datasets"]["name"] == "Data51":
        train_set = SLFDataset(image_dir=train_path,
                                image_size=opt["model"]["diffusion"]["image_size"], train_frac=opt["datasets"]
                               ["train"]["data_frac"], save_dir=opt["path"]["dataprocessor"], use_log_scale=opt["datasets"]["use_log_scale"])
    elif opt["datasets"]["name"] == "RadioMapSteerWOcarsDPM-625Buildings":
        train_set = RadioMapSeerSLFDataset(root_dir=train_path, gain_folder=opt["datasets"]["gain_folder"],
                                        building_mask_folder=opt["datasets"]["building_mask_folder"],
                                        car_mask_folder=opt["datasets"]["car_mask_folder"],
                                        road_mask_folder=opt["datasets"]["road_mask_folder"],
                                        image_size=opt["model"]["diffusion"]["image_size"],
                                        train_frac=opt["datasets"]["train"]["data_frac"],
                                        save_dir=opt["path"]["dataprocessor"])
    else:
        raise ValueError("Dataset not supported!!!")

    # train_loader = DataLoader(train_set, batch_size=16,
    #                           num_workers=4, shuffle=True)
    if opt["datasets"]["name"] == "RadioMapSteerWOcarsDPM-SamplingCondition-SLF-RMgen" or opt["datasets"]["name"] == "RadioMapSteerWOcarsDPM-SamplingCondition-RM" or opt["datasets"]["name"] == "RadioMapSteerWOcarsDPM-SamplingCondition-RM-fixC":
        batch_size = 16
    else:
        batch_size = 32
        
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=8, shuffle=True)

    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_SLF_model(opt)
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
                        train_images = train_data["SLF"].cpu()
                        train_images = train_images[:8] if train_images.size(
                            0) > 8 else train_images
                        log_images_as_grid(train_set.reverse_transform(
                            train_images), "Training Data")

                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    mat_result_path = '{}/{}'.format(opt['path']
                                                     ['mat_results'], current_epoch)
                    os.makedirs(mat_result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    val_images = []
                    condition_images = []
                    for idx in range(8):
                        if "mask" in train_data:
                            condition_x = train_data["mask"][idx:idx+1,:,:,:]
                        else:
                            condition_x = None

                        diffusion.sample(continous=False, condition_x=condition_x)
                        visuals = diffusion.get_current_visuals(sample=True)
                        sample_img = Metrics.tensor2img(
                            visuals['SAMPLE'])  # uint8
                        # rgbsample_img = Metrics.tensor2rgb_band8(
                        #     visuals['SAMPLE'])  # uint8

                        val_images.append(visuals['SAMPLE'].cpu().numpy())
                        if condition_x is not None:
                            condition_images.append(condition_x[0].cpu().numpy())

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                    if wandb_logger:
                        val_images = np.concatenate(val_images, axis=0)
                        log_images_as_grid(train_set.reverse_transform(
                            val_images), "Sampled Data")
                        if condition_images:
                            condition_images = np.concatenate(
                                condition_images, axis=0)
                            log_images_as_grid(condition_images, "Condition Data", plot_type="Condition", data_range=[condition_images.min(), condition_images.max()])

                # if current_step % opt['train']['save_checkpoint_freq'] == 0:
                if current_step % 25000 == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(
                            current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

        result_path = '{}'.format(opt['path']['results'])
        mat_result_path = '{}'.format(opt['path']['mat_results'])

        os.makedirs(result_path, exist_ok=True)
        os.makedirs(mat_result_path, exist_ok=True)

        sample_imgs = []
        for idx in range(40):
            idx += 1
            diffusion.sample(continous=True)
            visuals = diffusion.get_current_visuals(sample=True)

            show_img_mode = 'grid'
            if show_img_mode == 'single':
                # single img series
                sample_img = visuals['SAMPLE']  # uint8
                sample_num = sample_img.shape[0]
                for iter in range(0, sample_num):
                    visual.save_img(
                        visual.tensor2img(sample_img[iter]), '{}/{}_{}_sample_{}.png'.format(result_path, current_step, idx, iter))

            else:
                visual.save_img(
                    visual.tensor2rgb_band8(visuals['SAMPLE'][-1]), '{}/{}_{}_sample_preview.png'.format(result_path, current_step, idx))

                visual.save_mat(
                    visual.tensor2img(visuals['SAMPLE'][-1]), '{}/{}_{}_sample_abu.mat'.format(mat_result_path, current_step, idx))

            sample_imgs.append(visual.tensor2img(visuals['SAMPLE'][-1]))

        if wandb_logger:
            wandb_logger.log_images('eval_images', sample_imgs)
