import torch.nn as nn
import torch
import numpy as np
import math
from tqdm import tqdm
from .slf.networks import UNet
from .psd.networks import PSDNet
from .weight_initializer import init_weights
from inspect import isfunction
import wandb


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        data_point_shape,
        channels=3,
        loss_type='l1',
        conditional=True,
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.data_point_shape = data_point_shape
        self.loss_type = loss_type
        self.conditional = conditional

    
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError(f'Loss type {self.loss_type} not implemented')
    
    @staticmethod
    def warmup_beta_schedule(start, end, n_timestep, warmup_frac):
        '''Linear increase from start to end over warmup_steps, then constant at end'''
        betas = end * np.ones(n_timestep, dtype=np.float64)
        warmup_time_end = int(n_timestep * warmup_frac)
        betas[:warmup_time_end] = np.linspace(start, end, warmup_time_end, dtype=np.float64)
        return betas
        
    @staticmethod
    def get_beta_schedule(schedule, n_timestep, start, end, cosine_shift=8e-3):
        if schedule == 'linear':
            betas = np.linspace(start, end, n_timestep, dtype = np.float64)
        elif schedule == 'quad':
            betas = np.linspace(start**0.5, end**0.5, n_timestep, dtype = np.float64)**2
        elif schedule == 'warmup10':
            betas = GaussianDiffusion.warmup_beta_schedule(start, end, n_timestep, 0.1)
        elif schedule == 'warmup50':
            betas = GaussianDiffusion.warmup_beta_schedule(start, end, n_timestep, 0.5)
        elif schedule == 'const':
            betas = end * np.ones(n_timestep, dtype=np.float64)
        elif schedule == 'jst':
            '''1/T,1/(T-1),1/(T-2),...,1'''
            betas = 1.0 / np.linspace(end, start, n_timestep, dtype=np.float64)
        elif schedule == 'cosine': # View this paper "Noise Schedules for Diffusion Probabilistic Models" by Nikolas Kirschstein
            norm_timesteps = np.arange(n_timestep+1, dtype=np.float32) / n_timestep/n_timestep + cosine_shift
            ft = np.cos(math.pi/2 * norm_timesteps/(1+cosine_shift)).pow(2)
            alphas_bar = ft/ft[0]
            alphas = alphas_bar[1:]/ alphas_bar[:-1]
            betas = 1-alphas_bar
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(f'Schedule {schedule} not implemented')
        
        return betas
            
            
    def set_noise_schedule(self, schedule_opt, device):
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        
        '''Get betas schedule'''
        betas = self.get_beta_schedule(schedule=schedule_opt['schedule'],
                                       n_timestep = schedule_opt['n_timestep'],
                                       start=schedule_opt['linear_start'],
                                       end=schedule_opt['linear_end'])
        
        '''Compute required variables'''
        alphas = 1.0 - betas
        alphas_bar_t = np.cumprod(alphas, axis=0)
        alphas_bar_prev_t = np.append(1.0, alphas_bar_t[:-1])
        # self.sqrt_alphas_bar_t_prev = np.sqrt(np.append(1.0, alphas_bar_t)) # ?: Why all alphas_bar_t but t_prev in variable name
        
        self.num_timesteps = len(betas) 
        
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_bar_t', to_torch(alphas_bar_t))
        self.register_buffer('alphas_bar_prev_t', to_torch(alphas_bar_prev_t))
        
        # Variables for computation of diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar_t', to_torch(np.sqrt(alphas_bar_t)))
        self.register_buffer('sqrt_alphas_bar_prev_t', to_torch(np.sqrt(alphas_bar_prev_t)))
        self.register_buffer('sqrt_one_minus_alphas_bar_t', to_torch(np.sqrt(1-alphas_bar_t)))
        self.register_buffer('log_one_minus_alphas_bar_t', to_torch(np.log(1-alphas_bar_t)))
        self.register_buffer('sqrt_recip_alphas_bar_t', to_torch(np.sqrt(1/alphas_bar_t)))
        self.register_buffer('sqrt_recip_minus_1_alphas_bar_t', to_torch(np.sqrt(1/alphas_bar_t-1)))
        
        # Variables for posterior q(x_{t-1} | x_t, x_0)
        # From DDPM paper
        posterior_variance = betas * (1-alphas_bar_prev_t) / (1-alphas_bar_t) 
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas*np.sqrt(alphas_bar_prev_t) / (1-alphas_bar_t)))   
        self.register_buffer('posterior_mean_coef2', to_torch(np.sqrt(alphas)*(1-alphas_bar_prev_t)  / (1-alphas_bar_t)))  
    
    def set_ddim_steps(self, num_ddim_timesteps, type = 'linear'): # Take 20 ddim steps and 2000 diffusion steps; i.e. from 0 to 1999, as an example

        if self.num_timesteps % num_ddim_timesteps != 0:
            print("num_timesteps cannot be divisible by num_ddim_timesteps")

        if type == 'linear':
            steps = np.linspace(0, self.num_timesteps - 1, num_ddim_timesteps) # e.g., 0, 105.xx, ..., 1999
            ddim_timesteps = [int(step) for step in steps]
        elif type == 'cosine':
            # Cosine schedule for DDIM steps
            # More steps at the beginning (high noise), fewer at the end (low noise)
            t_normalized = np.linspace(0, 1, num_ddim_timesteps)
            # Apply cosine function to get non-linear spacing
            cosine_steps = (1 - np.cos(t_normalized * np.pi / 2)) * (self.num_timesteps - 1)
            ddim_timesteps = [int(step) for step in cosine_steps]
        else:
            raise NotImplementedError(f'DDIM step type {type} not implemented')
        
        self.ddim_timesteps = ddim_timesteps
        return ddim_timesteps
    
    def get_EDM_learning_rate(self, t, lambda_0 = 1e-4): # make the data consistent learning rate schedule: bigger in the middle, and smaller in the beginning and end

        n_timesteps = self.num_timesteps
        device = self.betas.device  # Get device from registered buffer
        lr_ratio = torch.ones(n_timesteps, dtype=torch.float32, device=device)
        
        lr_ratio[0:1000] = torch.linspace(1, 5.0, 1000, device=device)
        lr_ratio[1000:2000] = torch.linspace(5.0, 0.5, 1000, device=device)

        return lr_ratio[t] * lambda_0  # Index to get the LR for timestep t
    
    # Predicts x_0. x_{t-1} can be thought as predicted x_0 + noise at this timestep
    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_bar_t[t] * (x_t - self.sqrt_one_minus_alphas_bar_t[t] * noise)
    
    # '''Compute mean and variance of q(x_{t-1} | x_t, x_0) given x_0. Equation 7 in DDPM paper'''
    def q_posterior(self, x_start, x_t, t):
        mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        log_variance = self.posterior_log_variance_clipped[t]
        return mean, log_variance
    
    # Predict x_0 and then use it to compute the posterior mean and variance
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_bar_t[t]]).repeat(batch_size,1).to(x.device) # different than original code
        
        if condition_x is not None:
            predicted_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            predicted_noise = self.denoise_fn(x, noise_level)
        
        x_start = self.predict_start_from_noise(x_t=x, t=t, noise=predicted_noise)
        
        if clip_denoised:
            x_start = x_start.clamp(-1, 1)
        
        posterior_mean, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        
        return posterior_mean, posterior_log_variance
    
    def get_start_with_grad_enabled(self, x, t, clip_denoised=True, condition_x=None):
        x.requires_grad = True
        
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_bar_t[t]]).repeat(batch_size,1).view(-1,1,*([1]*len(self.data_point_shape))).to(x.device) # different than original code
        
        if condition_x is not None:
            predicted_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            predicted_noise = self.denoise_fn(x, noise_level)
        
        x_start = self.predict_start_from_noise(x_t=x, t=t, noise=predicted_noise)
        
        if clip_denoised:
            x_start = x_start.clamp(-1, 1)
        return x_start
    
    def get_start_with_grad_unspecify(self, x, t, clip_denoised=True, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_bar_t[t]]).repeat(batch_size,1).view(-1,1,*([1]*len(self.data_point_shape))).to(x.device) # different than original code
        
        if condition_x is not None:
            predicted_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            predicted_noise = self.denoise_fn(x, noise_level)
        
        x_start = self.predict_start_from_noise(x_t=x, t=t, noise=predicted_noise)
        
        if clip_denoised:
            x_start = x_start.clamp(-1, 1)
        return x_start
    
    def get_start_with_grad_disabled(self, x, t, clip_denoised=True, condition_x=None):
        # x.requires_grad = False
        
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_bar_t[t]]).repeat(batch_size,1).view(-1,1,*([1]*len(self.data_point_shape))).to(x.device) # different than original code
        
        if condition_x is not None:
            predicted_noise = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            predicted_noise = self.denoise_fn(x, noise_level)
        
        x_start = self.predict_start_from_noise(x_t=x, t=t, noise=predicted_noise)
        
        if clip_denoised:
            x_start = x_start.clamp(-1, 1)
        return x_start

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        posterior_mean, posterior_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = (0.5*posterior_log_variance).exp() * (torch.randn_like(x) if t>0 else torch.zeros_like(x))
        return posterior_mean + noise
    
    @torch.no_grad()
    def p_sample_loop(self, generation_shape, condition_x=None, continuous=False):
        device = self.betas.device
        sample_interval = (1 | (self.num_timesteps//10))
        
        
        x_T = torch.randn(generation_shape, device=device)
        ret_samples_xt = x_T
        
        x_t = x_T
        for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            if not self.conditional:
                x_t = self.p_sample(x_t, t)
            else:
                x_t = self.p_sample(x_t, t, condition_x=condition_x)
            
            if t % sample_interval == 0:
                ret_samples_xt = torch.concat([ret_samples_xt, x_t], dim=0)

        if continuous:
            return ret_samples_xt
        else:
            return ret_samples_xt[-1]
    
    def p_sample_step_inverse(self, x_t, t,  condition_x=None, lr_measurement_loss=1e-3):
        device = self.betas.device
        x_t = x_t.to(device)
        measurement_grad = torch.clamp(x_t.grad, min=-1e2, max=1e2) if x_t.grad is not None else 0
        # measurement_grad = x_t.grad if x_t.grad is not None else 0
        # wandb.log({'measurement_grad': measurement_grad.sum().item()})
        
        # '''Compute conditional grad (DPS)'''
        # measurement_grad = torch.autograd.grad(measurement_loss, x_t)[0]
        
        with torch.no_grad():
            # Todo: Enable denoising   
            if not self.conditional:
                x_t = self.p_sample(x_t, t)
            else:
                x_t = self.p_sample(x_t, t, condition_x=condition_x)
            
            '''Use the gradient of measurement loss to update x_t'''
            x_t = x_t - lr_measurement_loss * measurement_grad
        
        return x_t
        
    def p_sample_step_inverse_xule(self, x_t, t,  condition_x=None, lr_measurement_loss=1e-3):
        device = self.betas.device
        x_t = x_t.to(device)
        measurement_grad = torch.clamp(x_t.grad, min=-1e2, max=1e2) if x_t.grad is not None else 0
        
        # with torch.no_grad():
        #     # Todo: Enable denoising   
        #     if not self.conditional:
        #         x_t, noise = self.p_sample_without_noise(x_t, t)
        #     else:
        #         x_t, noise = self.p_sample_without_noise(x_t, t, condition_x=condition_x)
            
        #     '''Use the gradient of measurement loss to update x_t'''
        #     x_t = x_t - lr_measurement_loss * measurement_grad
        
        with torch.no_grad():
            if not self.conditional:
                posterior_mean, posterior_log_variance = self.p_mean_variance(x=x_t, t=t, clip_denoised=True, condition_x=None)
                noise = (0.5*posterior_log_variance).exp() * (torch.randn_like(x_t) if t>0 else torch.zeros_like(x_t))
                x_start = self.get_start_with_grad_enabled(x_t + noise, t, clip_denoised=True, condition_x=None)
            else:
                posterior_mean, posterior_log_variance = self.p_mean_variance(x=x_t, t=t, clip_denoised=True, condition_x=condition_x)
                noise = (0.5*posterior_log_variance).exp() * (torch.randn_like(x_t) if t>0 else torch.zeros_like(x_t))
                x_start = self.get_start_with_grad_enabled(x_t + noise, t, clip_denoised=True, condition_x=condition_x)
            
            noise_est = x_t - x_start
        
            diffusion_grad = torch.clamp(self.posterior_mean_coef1[t] * noise_est, min=-1e2, max=1e2)
            x_t = x_t - lr_measurement_loss * measurement_grad - diffusion_grad
        
        return x_t
    
    def p_sample_step_inverse_hqs(self, x_t, t, condition_x=None, lr_measurement_loss=1e-3):
        return x_t
    
    def noise_schedule_check(self, t):
        return self.sqrt_alphas_bar_t[t], self.sqrt_one_minus_alphas_bar_t[t]
        # self.sqrt_recip_alphas_bar_t[t], self.sqrt_one_minus_alphas_bar_t[t]
    
    @torch.no_grad()
    def p_sample_without_noise(self, x, t, clip_denoised=True, condition_x=None):
        posterior_mean, posterior_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = (0.5*posterior_log_variance).exp() * (torch.randn_like(x) if t>0 else torch.zeros_like(x))
        return posterior_mean, noise
        
    @torch.no_grad()
    def sample(self, batch_size=1, continuous=False, condition_x=None):
        feature_shape = self.data_point_shape
        channels = self.channels
        
        return self.p_sample_loop((batch_size, channels, *feature_shape), continuous=continuous, condition_x=condition_x)
    
    @torch.no_grad()
    def forward_sample(self, x_0, t_lst , noise): # different: not using continuous_sqrt_alpha_cumprod
        
        sqrt_alphas_bar_t = self.sqrt_alphas_bar_t[t_lst].to(x_0.device).view(-1,1,*([1]*len(self.data_point_shape)))
        
        x_t = sqrt_alphas_bar_t * x_0 + noise * torch.sqrt(1-sqrt_alphas_bar_t**2)
        return x_t
    
    def p_losses(self, x_in, noise=None, data_key='SLF', condition_key='mask'):
        x_0 = x_in[data_key]
        b,c = x_0.shape[:2]
        
        t_lst = np.random.randint( self.num_timesteps-1, size=(b,))
         # different than original code
        
        noise = default(noise, lambda: torch.randn_like(x_0))
        
        x_noisy = self.forward_sample(x_0=x_0, t_lst=t_lst, noise=noise)
        
        sqrt_alphas_bar_t = self.sqrt_alphas_bar_t[t_lst].to(x_0.device).view(-1,1,*([1]*len(self.data_point_shape)))
        if not self.conditional:
            predicted_noise = self.denoise_fn(x_noisy, sqrt_alphas_bar_t)
        else:
            condition_x = x_in[condition_key]
            predicted_noise = self.denoise_fn(torch.cat([condition_x, x_noisy], dim=1), sqrt_alphas_bar_t)
        
        loss = self.loss_func(noise, predicted_noise)
        
        return loss
    
    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
        
        
# Generator
def define_SLF_G(opt):
    model_opt = opt['model']
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    model = UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netG = GaussianDiffusion(
        model,
        data_point_shape=(model_opt['diffusion']['image_size'],model_opt['diffusion']['image_size']),
        channels=model_opt['diffusion']['channels'],
        loss_type=model_opt['diffusion']['loss_type'],    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG 

def define_RM_G(opt):
    return define_SLF_G(opt)       

def define_image_G(opt):
    return define_SLF_G(opt)
            

def define_PSD_G(opt):
    model_opt = opt['model']
    if ('norm_groups' not in model_opt['psdnet']) or model_opt['psdnet']['norm_groups'] is None:
        model_opt['psdnet']['norm_groups']=32
    model = PSDNet(
        in_channel=model_opt['psdnet']['in_channel'],
        out_channel=model_opt['psdnet']['out_channel'],
        norm_groups=model_opt['psdnet']['norm_groups'],
        inner_channel=model_opt['psdnet']['inner_channel'],
        channel_mults=model_opt['psdnet']['channel_multiplier'],
        attention_resolutions=model_opt['psdnet']['attn_res'],
        res_blocks=model_opt['psdnet']['res_blocks'],
        dropout=model_opt['psdnet']['dropout'],
        K=model_opt['diffusion']['K']
    )
    netG = GaussianDiffusion(
        model,
        data_point_shape=(model_opt['diffusion']['K'],),
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG        
            
            
        
        
        