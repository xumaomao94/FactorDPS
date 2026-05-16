import argparse
import torch
from torch.utils.checkpoint import checkpoint
from core import wandb_logger
import core.logger as Logger
from core.wandb_logger import WandbLogger
import diffusionmodel as Model
import logging
import numpy as np
from core.loaddata import RadioMapSeerRMDataset
from tqdm import tqdm
import wandb
import os
import time
import matplotlib.pyplot as plt
from core.metrics import compare_mssim
from utils.measurement import SamplingQuantizationWithDithering
from utils.losses import loglikelihood_quantization, loglikelihood_l2distance
from skimage.transform import resize


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_rate_to_fraction(sample_rate):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate > 100:
        raise ValueError("sample_rate must be a fraction in (0, 1] or a percentage in [1, 100]")
    return sample_rate / 100.0 if sample_rate >= 1 else sample_rate


def format_sample_rate(sample_rate):
    if sample_rate >= 1 and float(sample_rate).is_integer():
        return str(int(sample_rate))
    sample_rate_percent = sample_rate * 100 if sample_rate < 1 else sample_rate
    return f"{sample_rate_percent:g}"


def generate_testing_data(sc_dataset, opt):
    K = opt['band_number']
    num_emitters = opt['emitter_number']
    sample_frac = sample_rate_to_fraction(opt['sample_rate'])

    print(
        "Generating testing data "
        f"(R={num_emitters}, K={K}, image_size={opt['image_size']}, "
        f"sample_frac={sample_frac:g})"
    )
    data = sc_dataset.get_a_random_sample_with_spec(
        num_emitters=num_emitters,
        sample_frac=sample_frac,
        K=K,
    )

    required_keys = ['SLF', 'PSD', 'RM', 'mask']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Generated testing data is missing keys: {missing_keys}")

    expected_rm_shape = (opt['image_size'], opt['image_size'], K)
    if data['RM'].shape != expected_rm_shape:
        raise ValueError(f"Generated RM shape {data['RM'].shape} does not match expected {expected_rm_shape}")

    return data


def run_dps(data, sc_dataset, opt, psd_diffusion, slf_diffusion, wandb_logger, condition_key=None):

    # Start timing
    start_time = time.time()

    I,J,K = data['RM'].shape
    num_emitters = opt['emitter_number']

    '''Sample channel bands for visualization'''
    if K == 1:
        viz_rm_channels = [0]
    else:
        viz_rm_channels = np.random.choice(K, 3, replace=False)
        viz_rm_channels.sort()
    print(f"Visualizing Radio Map channels: {viz_rm_channels}")
    
    '''Initialize data'''
    device = slf_diffusion.netG.betas.device
    slf_t = torch.randn(num_emitters,1, I, J).to(device)
    if K == 1 and num_emitters == 1:
        psd_t = torch.ones(num_emitters,1, K).to(device)  # fix PSD for single emitter, single band case (slf = rm in this case)
    elif K == 1 and num_emitters > 1:
        psd_t = 0.5*torch.ones(num_emitters,1, K).to(device)
    else:
        psd_t = torch.randn(num_emitters,1, K).to(device)
    
    '''Enable grad for conditional grad of DPS'''
    slf_t.requires_grad = True
    if K == 1 and num_emitters == 1:
        psd_t = torch.ones_like(psd_t).to(device)  # fix PSD for single emitter, single band case (slf = rm in this case)
        psd_t.requires_grad = False
    elif K == 1 and num_emitters > 1:
        psd_t = 0.5*torch.ones_like(psd_t).to(device)
        psd_t.requires_grad = False
    else:
        psd_t.requires_grad = True
    
    assert psd_diffusion.netG.num_timesteps == slf_diffusion.netG.num_timesteps, "Number of time steps should be the same for PSD and SLF diffusion models"
    num_timesteps = psd_diffusion.netG.num_timesteps
    
    ground_truth_radio_map = torch.from_numpy(data['RM']).to(device).float()
    
    '''Setup measurement operator and loss function'''
    operator = SamplingQuantizationWithDithering(device=device,
                                                sample_frac=1.0, # we handle sampling mask outside the operator
                                                dithering_type="Inactive" if opt["dithering_sigma"] == 0 else "gaussian",
                                                dithering_sigma=opt["dithering_sigma"],
                                                quantization_type="Inactive" if opt["quantization_bits"] == 0 else "uniform",
                                                quantization_bit=opt["quantization_bits"],
                                                rm_min_value=-147, # -147, see dataset paper
                                                rm_max_value=-47, # -47, see dataset paper
                                                part_missing="hole" if opt["hole_ratio"] > 0 else None,
                                                hole_ratio= opt["hole_ratio"]
                                                )
    measurement, b_up, b_low, hole_mask = operator(sc_dataset._to_log(ground_truth_radio_map), return_boundaries=True)
    
    data_mask = data['mask'] if 'mask' in data and data['mask'] is not None else None
    sample_mask = hole_mask
    building_mask = None
    mask_tensor = None

    if data_mask is not None:
        '''combine the mask defined in the inner operator with that loaded from data'''
        sample_mask = data_mask[-1] != 0 # data_mask[-1]: store the sampled data
        sample_mask = sample_mask & hole_mask # the original sampling mask is stored in data['mask'][-1], and the sample_mask generated here is for hole_missing only
        # Extract building mask (first channel if multi-channel)
        building_mask = data_mask[0]
        mask_tensor = building_mask.astype(bool)
        sample_mask = sample_mask & ~mask_tensor # building_mask = 1 => there is building
        measurement[~sample_mask, :] = 0

    observed_count = int(np.sum(sample_mask))
    total_count = int(sample_mask.size)
    print(f"Observation samples after masks: {observed_count}/{total_count} ({observed_count / total_count:.4%})")
    if observed_count == 0:
        print("Warning: no observable points remain after applying sampling, building, and hole masks.")

    if opt["quantization_bits"] != 0:
        loss = lambda inferred_x: loglikelihood_quantization(inferred_x, sample_mask=sample_mask, sigma=1, b_up=b_up, b_low=b_low)
    else:
        if building_mask is not None:
            loss = lambda inferred_x: loglikelihood_l2distance(torch.clamp(inferred_x, min = -147), sample_mask=sample_mask, measurement=torch.clamp(measurement, min = -147), sigma = 1)
        else:
            loss = lambda inferred_x: loglikelihood_l2distance(inferred_x, sample_mask=sample_mask, measurement=measurement, sigma = 1)

    '''Conditioning'''
    condition_x = torch.from_numpy(data[condition_key][0]).repeat(num_emitters,1,1,1).to(device).float() if condition_key is not None and condition_key in data else None
    if condition_x is not None:
        print(f"Conditioning with {condition_key}")
    else:
        print("No conditioning")
    
    for t in tqdm(reversed(range(num_timesteps)), desc="Inverse Sampling", total=num_timesteps):

        '''Predicted expected start and compute dps measurement loss'''
        slf_start = slf_diffusion.netG.get_start_with_grad_enabled(slf_t, t, condition_x=condition_x)
        if K == 1 and num_emitters == 1:
            psd_start = psd_t  # fix PSD for single emitter, single band case (slf = rm in this case)
            # Use psd_t directly to ensure gradient flow, not psd_start
            inferred_radio_map = sc_dataset.compute_radio_map(sc_dataset.get_original_scale_slf(slf_start), sc_dataset.get_original_scale_psd(psd_start))
        elif K == 1 and num_emitters > 1:
            psd_start = psd_t  # fix PSD for multi emitter, single band case (slf = rm in this case)
            inferred_radio_map = sc_dataset.compute_radio_map(sc_dataset.get_original_scale_slf(slf_start), psd_start) # psd = 0.5 in this case
        else:
            psd_start = psd_diffusion.netG.get_start_with_grad_enabled(psd_t, t)
            inferred_radio_map = sc_dataset.compute_radio_map(sc_dataset.get_original_scale_slf(slf_start), sc_dataset.get_original_scale_psd(psd_start))
        
        loss_val = loss(sc_dataset._to_log(inferred_radio_map))
        loss_val.backward()
        
        '''Run denoising'''
        slf_dps_lr = opt['conditioning']['slf_dps_lr']
        psd_dps_lr = opt['conditioning']['psd_dps_lr']
        
        # get EDM-consistent learning rate
        slf_dps_lr = slf_diffusion.netG.get_EDM_learning_rate(t, lambda_0=slf_dps_lr)
        psd_dps_lr = psd_diffusion.netG.get_EDM_learning_rate(t, lambda_0=psd_dps_lr)

        slf_t = slf_diffusion.netG.p_sample_step_inverse(slf_t, t, lr_measurement_loss=slf_dps_lr, condition_x=condition_x)
        if K == 1 and num_emitters == 1:
            psd_t = psd_t  # fix PSD for single emitter, single band case (slf = rm in this case)
        elif K == 1: # no diffusion model needed for psd_t
            psd_t = psd_t
        else:
            psd_t = psd_diffusion.netG.p_sample_step_inverse(psd_t, t, lr_measurement_loss=psd_dps_lr)

        '''Zero out grad'''
        slf_t.grad = None
        if K != 1 or num_emitters != 1:
            psd_t.grad = None
        
        '''Evaluation of Radio map'''
        gt_rm = sc_dataset._to_log(data['RM'])
        inferred_rm = sc_dataset._to_log(inferred_radio_map.detach().cpu().numpy())
        
        if mask_tensor is not None:
            gt_rm[mask_tensor, :] = -147
            inferred_rm[mask_tensor, :] = -147
        lnrse_t = np.sum((gt_rm-inferred_rm)**2)/np.sum(gt_rm**2)

    slf_final = sc_dataset.get_original_scale_slf(slf_t.detach().squeeze().cpu().numpy())
    if slf_final.ndim == 2:
        slf_final = slf_final[np.newaxis,:,:] # [num_emitters, H, W]
    psd_final = sc_dataset.get_original_scale_psd(psd_t.detach().squeeze().cpu().numpy()) if K > 1 else psd_t.detach().squeeze().cpu().numpy()
    if psd_final.ndim == 1:
        psd_final = psd_final[np.newaxis,:]  # [num_emitters, K]
    rm_final = inferred_radio_map.detach().squeeze().cpu().numpy()
    if rm_final.ndim == 2:
        rm_final = rm_final[:,:,np.newaxis]  # [I, J, K]
    rm_final_log = sc_dataset._to_log(rm_final)
    mask_final = data['mask'].copy() if data_mask is not None else None
    if mask_final is not None:
        mask_final[-1] = sample_mask.astype(np.uint8)

    # Calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time

    save_folder = opt.get('save_folder') or 'results'
    os.makedirs(save_folder, exist_ok=True)
    save_name = opt.get('save_name')
    if save_name is None:
        save_name = os.path.join(save_folder, f"{opt['method']}_exp{opt['exp_id']:03d}.npz")
    else:
        save_name = os.path.join(save_folder, os.path.basename(save_name))
    opt['save_name'] = save_name
    
    gt_rm_log = sc_dataset._to_log(data['RM'])
    rm_final_log_eval = rm_final_log.copy()
    if mask_tensor is not None:
        gt_rm_log[mask_tensor, :] = -147
        rm_final_log_eval[mask_tensor, :] = -147

    gt_rm_mssim = gt_rm_log
    if gt_rm_log.ndim == 2:
        gt_rm_mssim = gt_rm_log[:,:,np.newaxis]
    rm_final_log_mssim = rm_final_log_eval
    if rm_final_log_eval.ndim == 2:
        rm_final_log_mssim = rm_final_log_eval[:,:,np.newaxis]
    
    mssim_val = compare_mssim(gt_rm_mssim, rm_final_log_mssim, data_range=100, multidimension=False)

    np.savez_compressed(save_name, SLF=slf_final, PSD=psd_final, RM=rm_final, RM_log=rm_final_log, mask=mask_final,
                       lnrse=lnrse_t, mssim=mssim_val, runtime_seconds=total_runtime, measurement=measurement.detach().cpu().numpy())

    print("Inverse Sampling Done")
    print(f"Final LNRSE: {lnrse_t}")
    print(f"Final MSSIM: {mssim_val}")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    
    measurement_log = measurement.detach().cpu().numpy()
    if measurement_log.ndim == 2:
        measurement_log = measurement_log[:,:,np.newaxis]
    sample_mask_plot = np.asarray(sample_mask, dtype=bool)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='lightgray')
    fig, axes = plt.subplots(3, K, figsize=(5*K, 12), squeeze=False)
    for k in range(K):
        observation_image = np.full_like(measurement_log[:, :, k], np.nan, dtype=float)
        observation_image[sample_mask_plot] = measurement_log[:, :, k][sample_mask_plot]
        panels = [
            ("Ground Truth", gt_rm_mssim[:, :, k]),
            (f"Observation ({observed_count} samples)", np.ma.masked_invalid(observation_image)),
            ("Recovered", rm_final_log_mssim[:, :, k]),
        ]
        for row, (title, image) in enumerate(panels):
            im = axes[row, k].imshow(image, cmap=cmap, vmin=-147, vmax=-47, interpolation='nearest')
            axes[row, k].set_title(f"{title} - Band {k}")
            axes[row, k].axis('off')
            plt.colorbar(im, ax=axes[row, k], fraction=0.046, pad=0.04)

    fig.suptitle(f"LNRSE: {lnrse_t:.4f}, MSSIM: {mssim_val:.4f}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    result_base = os.path.splitext(os.path.basename(save_name))[0]
    save_path = os.path.join(save_folder, f"{result_base}_visualization.png")
    plt.savefig(save_path)
    plt.close()

    return lnrse_t, inferred_radio_map.detach().squeeze().cpu().numpy(), slf_t.detach().squeeze().cpu().numpy(), psd_t.detach().squeeze().cpu().numpy() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    '''Method settings'''
    parser.add_argument('-c','--config', type=str, default='config/inference.json')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true', default=False)
    parser.add_argument('-enable_wandb', action='store_true', default=False)
    parser.add_argument('-num_runs', type=int, default=1)
    parser.add_argument('-method', type=str, default='dps', choices=['dps', 'dm-plug', 'rm-gen', 'rm-gen-slf', 'dps-rm-gen', 'dps-rm-gen-uncondition'])

    '''Radio map settings'''
    parser.add_argument('-testing_data_folder', type=str, default='/nfs/stak/users/xul2/hpc-share/sc-diffuse/TSP/settings/exp1_mr')
    parser.add_argument('-testing_data_name', type=str, default=None)
    parser.add_argument('-exp_id',type=int,default=0)
    parser.add_argument('-load_testing_data', action='store_true', default=False, help='load legacy .npz testing data instead of generating it')
    
    parser.add_argument('-data_root_dir', type=str, default=None, help='RadioMapSeer root dir used to generate testing data')
    parser.add_argument('-image_size', type=int, default=256, help='generated radio map height/width; defaults to the SLF config image_size')
    parser.add_argument('-band_number', type=int, default=1)
    parser.add_argument('-emitter_number', type=int, default=1)

    
    '''Measurement settings'''
    parser.add_argument('-sample_rate', type=float, default=0.05) # fraction in (0, 1] or percentage in [1, 100]
    parser.add_argument('-quantization_bits',type=int, default=0) # 0 means no quantization
    parser.add_argument('-dithering_sigma',type=float,default=0.0) # 0 means no dithering
    parser.add_argument('-hole_ratio', type=float, default=0.0) # middle hole missing, hole size: [M*hole_ratio, N*hole_ratio]
    parser.add_argument('-random_seed', type=int, default=0)
    
    '''Save settings'''
    parser.add_argument('-save_folder', type=str, default='experiments/demo', help='folder to save results')

    '''extract the arguments'''
    args = parser.parse_args()
    args.phase = "inference"
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Override config file values with command-line arguments
    opt['method'] = args.method
    opt['testing_data_folder'] = args.testing_data_folder
    opt['testing_data_name'] = args.testing_data_name
    opt['data_root_dir'] = args.data_root_dir
    opt['load_testing_data'] = args.load_testing_data
    opt['emitter_number'] = args.emitter_number
    opt['band_number'] = args.band_number
    opt['sample_rate'] = args.sample_rate
    opt['quantization_bits'] = args.quantization_bits
    opt['dithering_sigma'] = args.dithering_sigma
    opt['hole_ratio'] = args.hole_ratio
    opt['random_seed'] = args.random_seed
    opt['exp_id'] = args.exp_id
    opt['save_folder'] = args.save_folder
    opt['image_size'] = args.image_size
    
    psd_opt = opt['PSD_config']['config']
    psd_opt = Logger.dict_to_nonedict(psd_opt)
    slf_opt = opt['SLF_config']['config']
    slf_opt = Logger.dict_to_nonedict(slf_opt)
    if args.data_root_dir is not None:
        slf_opt['datasets']['data_root_dir'] = args.data_root_dir


    seed = opt['random_seed'] + opt['exp_id']
    set_random_seed(seed)
    print(f"Random seed set to {seed}")
    
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

    '''Setup logger'''
    Logger.setup_logger(opt['method'], opt['path']['log'],
                        'inference', level=logging.INFO, screen=False)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    sample_rate_label = format_sample_rate(opt['sample_rate'])
    if opt['emitter_number'] == 1:
        opt['wandb']['run_name'] = f"{opt['method']}-sr{sample_rate_label}-bits{int(opt['quantization_bits'])}-hole{int(opt['hole_ratio']*100)}-exp{opt['exp_id']:03d}"
    else:
        opt['wandb']['run_name'] = f"{opt['method']}-R{opt['emitter_number']}-sr{sample_rate_label}-bits{int(opt['quantization_bits'])}-hole{int(opt['hole_ratio']*100)}-exp{opt['exp_id']:03d}"       
    # Initialize WandbLogger
    wandb_logger = WandbLogger(opt) if args.enable_wandb else None
    if opt['emitter_number'] == 1:
        opt['save_name'] = os.path.join(opt['save_folder'], f"{opt['method']}_sr{sample_rate_label}_bits{int(opt['quantization_bits'])}_hole{int(opt['hole_ratio']*100)}_exp{opt['exp_id']:03d}.npz")
    else:
        opt['save_name'] = os.path.join(opt['save_folder'], f"{opt['method']}_R{opt['emitter_number']}_sr{sample_rate_label}_bits{int(opt['quantization_bits'])}_hole{int(opt['hole_ratio']*100)}_exp{opt['exp_id']:03d}.npz")
    
    '''Radio Map dataset'''
    I,J = opt['image_size'], opt['image_size']
    K = opt['band_number']
    if not (K == 1 or K == psd_opt['datasets']['K']):
        raise ValueError("band_number must be 1 or match the PSD model K")
    num_emitters = opt['emitter_number']
    
    '''load dataset module'''
    dataset_name = slf_opt['datasets']['name']
    if dataset_name == "RadioMapSteerWOcarsDPM-625Buildings":
        sc_dataset = RadioMapSeerRMDataset(slf_root_dir=slf_opt['datasets']['data_root_dir'],
                                            num_peaks_per_psd=psd_opt['datasets']['num_peaks'],
                                            K=opt['band_number'],
                                            basis_type=psd_opt['datasets']['basis_type'],
                                            image_size=opt['image_size'],
                                            slf_metadata_dir=opt['SLF_config']['dataprocessor'],
                                            psd_metadata_dir=opt['PSD_config']['dataprocessor'],
                                            slf_folder = slf_opt['datasets']['gain_folder'],
                                            building_mask_folder=slf_opt["datasets"]["building_mask_folder"],
                                            car_mask_folder=slf_opt["datasets"]["car_mask_folder"],
                                            road_mask_folder=slf_opt["datasets"]["road_mask_folder"],
                                            sampling_conditional=True,
                                            )
        condition_key = 'mask' if slf_opt["model"]["diffusion"]["conditional"] else None
    elif dataset_name == "RadioMapSteerWOcarsDPM-SamplingCondition" or dataset_name == "RadioMapSteerWOcarsDPM-SamplingCondition-SLF" or dataset_name == "RadioMapSteerWOcarsDPM-SamplingCondition-RM" or dataset_name == "RadioMapSteerWOcarsDPM-625Buildings-RM":
        sc_dataset = RadioMapSeerRMDataset(slf_root_dir=slf_opt['datasets']['data_root_dir'],
                                           num_peaks_per_psd=psd_opt['datasets']['num_peaks'],
                                            K=opt['band_number'],
                                            basis_type=psd_opt['datasets']['basis_type'],
                                            image_size=opt['image_size'],
                                            slf_metadata_dir=opt['SLF_config']['dataprocessor'],
                                            psd_metadata_dir=opt['PSD_config']['dataprocessor'],
                                            slf_folder = slf_opt['datasets']['gain_folder'],
                                            building_mask_folder=slf_opt["datasets"]["building_mask_folder"],
                                            car_mask_folder=slf_opt["datasets"]["car_mask_folder"],
                                            road_mask_folder=slf_opt["datasets"]["road_mask_folder"],
                                            sampling_conditional=True,
                                            )
        condition_key = 'mask' if slf_opt["model"]["diffusion"]["conditional"] else None
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    '''load diffusion models'''
    psd_diffusion = Model.create_PSD_model(psd_opt)
    slf_diffusion = Model.create_SLF_model(slf_opt)
    
    '''Set noise schedule'''
    psd_diffusion.set_new_noise_schedule(
                        psd_opt['model']['beta_schedule']['val'], schedule_phase='val')
    slf_diffusion.set_new_noise_schedule(
                        slf_opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    
    '''Prepare testing data'''
    if opt['load_testing_data']:
        if opt['testing_data_name'] is None:
            raise ValueError("testing_data_name is required when -load_testing_data is set")
        data_file = os.path.join(opt['testing_data_folder'], opt['testing_data_name']) # testing_data_name format: sr1-exp001.npz; 1 denote 1% sampling rate, 001 denote the run number
        data = np.load(data_file)   # keys: '{'SLF': slf, 'PSD': psd, 'RM': sc, 'mask': masks}
                                    # SLF: (num_emitters, I, J), data domain, 1e-127 ~ 1e-47
                                    # PSD: (num_emitters, K)
                                    # RM: (I, J, K), data domain, 1e-147 ~ 1e-37
                                    # masks: (num_layers, I, J), layer0: building mask, layer-1: sampling mask
    else:
        data = generate_testing_data(sc_dataset, opt)

    '''Run the methods'''
    if opt['method'] == 'dps':
        run_dps(data, sc_dataset, opt, psd_diffusion, slf_diffusion, wandb_logger, condition_key=condition_key)
    else:
        raise NotImplementedError(f"Method {opt['method']} not implemented")
                                
