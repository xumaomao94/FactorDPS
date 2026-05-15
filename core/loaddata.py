import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
from core import utils
import cv2
from tqdm import tqdm
from torchvision.transforms.v2 import Resize
from core.datasets import RadioMapSeerRMDataset
from datautils.psd_generate import PSDGenerator
from imageio.v3 import imread 
def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


class SCDataset:
    def __init__(self, slf_dir, num_peaks_per_psd, K, basis_type, slf_metadata_dir, psd_metadata_dir, image_size=None):
        
        '''Initialize PSD Generator'''
        self.num_peaks_per_psd = num_peaks_per_psd
        self.K = K
        self.basis_type = basis_type
        self.image_size = image_size
        self.psd_generator = PSDGenerator(num_peaks_per_psd=self.num_peaks_per_psd, K=self.K, basis_type=self.basis_type, seed=100)
        
        self.slf_dir = slf_dir
        self.slf_metadata_dir = slf_metadata_dir
        self.psd_metadata_dir = psd_metadata_dir
        
        '''Setup'''
        self.__setup__()
        
    def __setup__(self):
        '''Initialize SCF Generator'''
        self.image_folders = os.listdir(self.slf_dir)        
        self.image_files = []
        for i in self.image_folders:
            if is_mat_file(i):
                full_path = os.path.join(self.slf_dir, i)
                self.image_files.append(full_path)
        
        '''Load metadata'''
        self.slf_min, self.slf_max = np.load(os.path.join(self.slf_metadata_dir, "slf_min_max_values.npy"))
        self.psd_min, self.psd_max = np.load(os.path.join(self.psd_metadata_dir, "psd_min_max_values.npy"))
        self.slf_train_images = np.load(os.path.join(self.slf_metadata_dir, "slf_train_files.npy"))
        self.slf_test_images = sorted(list(set(self.image_files) - set(self.slf_train_images)))
        
        '''Compute min and max values for radio map from samples'''
        self.rm_min, self.rm_max = self.get_min_max_radio_map_values()
    
    def get_min_max_radio_map_values(self, sample_num=1000):
        '''Get min and max values for radio map'''
        rm_min, rm_max = np.inf, -np.inf
        for _ in tqdm(range(sample_num), desc="Computing min and max values for radio map", total=sample_num):
            if self.K == 1:
                sample = self.get_a_random_slf(1)
            else:
                sample = self.get_a_random_sample(1) # why is here (number_of_emitters=1)?
            radio_map=sample['RM']
            rm_min = min(radio_map.min(), rm_min)
            rm_max = max(radio_map.max(), rm_max)
        return rm_min, rm_max
        
    @staticmethod
    def _to_log(x):
        if torch.is_tensor(x):
            return 10*torch.log10(torch.clip(x, min=0, max=None)+1e-10)
        else:
            return 10*np.log10(np.clip(x, a_min=0, a_max=None)+1e-10)
    
    @staticmethod
    def _from_log(x):
        return 10**((x)/10.0)-1e-10
    
    def get_original_scale_rm(self, rm, revert_log=True):
        if revert_log:
            rm = self._from_log(rm)
            
        if torch.is_tensor(rm):
            return torch.clamp(rm, min=0, max=None)
        else:
            return np.clip(rm, a_min=0, a_max=None)
    
    
    def get_original_scale_slf_no_clip(self, slf, revert_log=True):
        slf = slf*(self.slf_max-self.slf_min)+self.slf_min
        if revert_log:
            slf = self._from_log(slf)
        return slf
        
        
    def get_original_scale_slf(self, slf, revert_log=True):
        slf = slf*(self.slf_max-self.slf_min)+self.slf_min
        if torch.is_tensor(slf):
            slf = torch.clamp(slf, min=self.slf_min, max=None)
        else:
            slf = np.clip(slf, a_min=self.slf_min, a_max=None)
        
        if revert_log:
            slf = self._from_log(slf)
            
        return slf
        

    
    def get_original_scale_psd_no_clip(self, psd):
        psd = psd*(self.psd_max-self.psd_min)+self.psd_min
        return psd
    
    def get_original_scale_psd(self, psd):
        psd = psd*(self.psd_max-self.psd_min)+self.psd_min
        
        if torch.is_tensor(psd):
            return torch.clamp(psd, min=self.psd_min, max=None)
        else:
            return np.clip(psd, a_min=self.psd_min, a_max=None)
    
    @staticmethod
    def compute_radio_map(slf, psd):
        
        if not torch.is_tensor(slf):
            slf = torch.from_numpy(slf)
            is_slf_numpy = True
        else:
            is_slf_numpy = False
        if not torch.is_tensor(psd):
            psd = torch.from_numpy(psd)
            is_psd_numpy = True
        else:
            is_psd_numpy = False
        
        if len(slf.shape) == 4:
            slf = slf.squeeze(1)
        if len(psd.shape) == 3:
            psd = psd.squeeze(1)
            
        assert slf.shape[0] == psd.shape[0], "Number of SLF and PSD should be the same"
        assert len(slf.shape) == 3, "SLF should be 3D"
        assert len(psd.shape) == 2, "PSD should be 2D"
        
        slf = slf.unsqueeze(dim=-1) # (num_emitters, H, W, 1)
        psd = psd.unsqueeze(dim=1).unsqueeze(dim=2) # (num_emitters, 1, 1, K)
        radio_map = torch.sum(slf * psd, dim=0) # (H, W, K)
        
        if is_slf_numpy and is_psd_numpy:
            radio_map = radio_map.numpy() # (H, W, K)
        return radio_map
        
    def get_a_random_sample(self, num_emitters, train=False):
        if train:
            candidate_slf_files = self.slf_train_images
        else:
            candidate_slf_files = self.slf_test_images
            
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_dir = candidate_slf_files[slf_index]
            data = sio.loadmat(load_dir)
            gt = np.array(data['Sc'][...], dtype=np.float32)
            if self.image_size is not None:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        psd = self.psd_generator.generate_psd(num_emitters).T # (num_emitters, K)
        
        sc = self.compute_radio_map(slf, psd)
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc}
            
    def get_a_random_slf(self, num_emitters, train=False):
        if train:
            candidate_slf_files = self.slf_train_images
        else:
            candidate_slf_files = self.slf_test_images
            
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_dir = candidate_slf_files[slf_index]
            data = sio.loadmat(load_dir)
            gt = np.array(data['Sc'][...], dtype=np.float32)
            if self.image_size is not None:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        psd = np.ones((num_emitters,self.K)) # (num_emitters, K)
        
        sc = self.compute_radio_map(slf, psd)
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc}
    
class RadioMapSeerSCDataset(SCDataset):
    def __init__(self, slf_root_dir, slf_folder,  num_peaks_per_psd, K, basis_type, slf_metadata_dir, psd_metadata_dir, image_size=None, building_mask_folder=None, car_mask_folder=None, road_mask_folder=None, sampling_conditional=False):
        self.slf_root_dir = slf_root_dir
        slf_dir = os.path.join(slf_root_dir,"gain", slf_folder)
        
        '''Directories for the condition'''
        self.building_mask_dir = os.path.join(self.slf_root_dir, "png", building_mask_folder) if building_mask_folder else None
        self.car_mask_dir = os.path.join(self.slf_root_dir, "png", car_mask_folder) if car_mask_folder else None
        self.road_mask_dir = os.path.join(self.slf_root_dir, "png", road_mask_folder) if road_mask_folder else None
        self.is_sampling_mask = sampling_conditional
        if self.is_sampling_mask:
            self.sampling_rate_range = [0.01,0.1]
            
        super(RadioMapSeerSCDataset, self).__init__(slf_dir, num_peaks_per_psd, K, basis_type, slf_metadata_dir, psd_metadata_dir, image_size)
        
        
       
    def __setup__(self):
        '''Scaling Factor for RadioMapSeer Dataset'''
        self.PL_min, self.PL_max = -186.0, -47.0
        
        
        '''Initialize SCF Generator'''
        self.image_folders = os.listdir(self.slf_dir)        
        self.image_files = []
        for i in self.image_folders:
            if  is_png_file(i):
                full_path = os.path.join(self.slf_dir, i)
                self.image_files.append(full_path)
        
        '''Load metadata'''
        self.slf_min, self.slf_max = np.load(os.path.join(self.slf_metadata_dir, "slf_min_max_values.npy"))
        self.psd_min, self.psd_max = np.load(os.path.join(self.psd_metadata_dir, "psd_min_max_values.npy"))
        self.slf_train_images = np.load(os.path.join(self.slf_metadata_dir, "slf_train_files.npy"))
        self.slf_test_images = sorted(list(set(self.image_files) - set(self.slf_train_images)))
        
        '''Separate train and test images based on scenario_id'''
        self.scenario_based_slf_train_images = dict()
        self.scenario_based_slf_test_images = dict()
        for file in self.slf_train_images:
            scenario_id = self.get_scenario_id(file)
            if scenario_id in self.scenario_based_slf_train_images:
                self.scenario_based_slf_train_images[scenario_id].append(file)
            else:
                 self.scenario_based_slf_train_images[scenario_id] = [file]
        
        for file in self.slf_test_images:
            scenario_id = self.get_scenario_id(file)
            if scenario_id in self.scenario_based_slf_test_images:
                self.scenario_based_slf_test_images[scenario_id].append(file)
            else:
                 self.scenario_based_slf_test_images[scenario_id] = [file]
        
        '''Compute min and max values for radio map from samples'''
        self.rm_min, self.rm_max = self.get_min_max_radio_map_values()
    
    @staticmethod
    def _to_log(x):
        if torch.is_tensor(x):
            return 10*torch.log10(torch.clip(x, min=0, max=None)+1e-30)
        else:
            return 10*np.log10(np.clip(x, a_min=0, a_max=None)+1e-30)
    
    @staticmethod
    def _from_log(x):
        return 10**((x)/10.0)-1e-30
    
    @staticmethod
    def get_scenario_id(file):
        return file.split('/')[-1].split(".png")[0].split("_")[0]
    
    def read_slf(self, file, db_scale=False):
        slf = imread(file)
        slf = (slf/255.0) * (self.PL_max - self.PL_min) + self.PL_min
        if not db_scale:
            slf = self._from_log(slf)
        return slf
    
    def get_a_random_sample(self, num_emitters, train=False):
        if train:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_train_images.keys())) if len(self.scenario_based_slf_train_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_train_images[scenario_id])
        else:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_test_images.keys())) if len(self.scenario_based_slf_test_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_test_images[scenario_id])
        
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_file_path = candidate_slf_files[slf_index]
            gt = self.read_slf(load_file_path)
            if self.image_size is not None and self.image_size != gt.shape[0]:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        psd = self.psd_generator.generate_psd(num_emitters).T # (num_emitters, K)
        
        sc = self.compute_radio_map(slf, psd)
        
        
        '''Load conditioning masks'''
        masks = []
        if self.building_mask_dir:
            building_mask = imread(os.path.join(self.building_mask_dir, f"{scenario_id}.png"))
            building_mask = building_mask/ 255.0
            masks.append(building_mask)
        if self.car_mask_dir:
            car_mask = imread(os.path.join(self.car_mask_dir, f"{scenario_id}.png"))
            car_mask = car_mask/255.0
            masks.append(car_mask)
        if self.road_mask_dir:
            road_mask = imread(os.path.join(self.road_mask_dir, f"{scenario_id}.png"))
            road_mask = road_mask/255.0
            masks.append(road_mask)
        if self.is_sampling_mask:
            sampling_rate = np.random.uniform(*self.sampling_rate_range)
            sampling_mask = np.random.binomial(1, sampling_rate, size=[self.image_size, self.image_size])
            sampled_gt = gt.squeeze() * sampling_mask
            masks.append(sampled_gt)
        if masks:
            masks = np.stack(masks, axis=0)
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(torch.from_numpy(masks)).numpy()
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc, 'mask': masks}
    
    
    def get_a_random_slf(self, num_emitters, train=False):
        if train:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_train_images.keys())) if len(self.scenario_based_slf_train_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_train_images[scenario_id])
        else:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_test_images.keys())) if len(self.scenario_based_slf_test_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_test_images[scenario_id])
        
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_file_path = candidate_slf_files[slf_index]
            gt = self.read_slf(load_file_path)
            if self.image_size is not None and self.image_size != gt.shape[0]:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        psd = np.ones((num_emitters,self.K)) # (num_emitters, K)
        
        sc = self.compute_radio_map(slf, psd)
        
        
        '''Load conditioning masks'''
        masks = []
        if self.building_mask_dir:
            building_mask = imread(os.path.join(self.building_mask_dir, f"{scenario_id}.png"))
            building_mask = building_mask/ 255.0
            masks.append(building_mask)
        if self.car_mask_dir:
            car_mask = imread(os.path.join(self.car_mask_dir, f"{scenario_id}.png"))
            car_mask = car_mask/255.0
            masks.append(car_mask)
        if self.road_mask_dir:
            road_mask = imread(os.path.join(self.road_mask_dir, f"{scenario_id}.png"))
            road_mask = road_mask/255.0
            masks.append(road_mask)
        if self.is_sampling_mask:
            sampling_rate = np.random.uniform(*self.sampling_rate_range)
            sampling_mask = np.random.binomial(1, sampling_rate, size=[self.image_size, self.image_size])
            sampled_gt = gt.squeeze() * sampling_mask
            masks.append(sampled_gt)
        if masks:
            masks = np.stack(masks, axis=0)
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(torch.from_numpy(masks)).numpy()
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc, 'mask': masks}
    
class RadioMapSeerRMDataset(SCDataset):
    def __init__(self, slf_root_dir, slf_folder,  num_peaks_per_psd, K, basis_type, slf_metadata_dir, psd_metadata_dir, image_size=None, building_mask_folder=None, car_mask_folder=None, road_mask_folder=None, sampling_conditional=False):
        self.slf_root_dir = slf_root_dir
        slf_dir = os.path.join(slf_root_dir,"gain", slf_folder)
        
        '''Directories for the condition'''
        self.building_mask_dir = os.path.join(self.slf_root_dir, "png", building_mask_folder) if building_mask_folder else None
        self.car_mask_dir = os.path.join(self.slf_root_dir, "png", car_mask_folder) if car_mask_folder else None
        self.road_mask_dir = os.path.join(self.slf_root_dir, "png", road_mask_folder) if road_mask_folder else None
        self.is_sampling_mask = sampling_conditional
        if self.is_sampling_mask:
            self.sampling_rate_range = [0.01,0.1]

        super(RadioMapSeerRMDataset, self).__init__(slf_dir, num_peaks_per_psd, K, basis_type, slf_metadata_dir, psd_metadata_dir, image_size)

    def __setup__(self):
        '''Scaling Factor for RadioMapSeer Dataset'''
        self.PL_min, self.PL_thre, self.PL_max = -147.0, -127.0, -47.0
        
        
        '''Initialize SCF Generator'''
        self.image_folders = os.listdir(self.slf_dir)        
        self.image_files = []
        for i in self.image_folders:
            if  is_png_file(i):
                full_path = os.path.join(self.slf_dir, i)
                self.image_files.append(full_path)
        
        '''Load metadata'''
        self.slf_min, self.slf_max = -127, -47 # np.load(os.path.join(self.slf_metadata_dir, "slf_min_max_values.npy"))
        # print(self.slf_min, self.slf_max)
        self.psd_min, self.psd_max = np.load(os.path.join(self.psd_metadata_dir, "psd_min_max_values.npy"))
        self.slf_train_images = np.load(os.path.join(self.slf_metadata_dir, "slf_train_files.npy"))
        self.slf_test_images = sorted(list(set(self.image_files) - set(self.slf_train_images)))
        
        '''Separate train and test images based on scenario_id'''
        self.scenario_based_slf_train_images = dict()
        self.scenario_based_slf_test_images = dict()
        for file in self.slf_train_images:
            scenario_id = self.get_scenario_id(file)
            if scenario_id in self.scenario_based_slf_train_images:
                self.scenario_based_slf_train_images[scenario_id].append(file)
            else:
                 self.scenario_based_slf_train_images[scenario_id] = [file]
        
        for file in self.slf_test_images:
            scenario_id = self.get_scenario_id(file)
            if scenario_id in self.scenario_based_slf_test_images:
                self.scenario_based_slf_test_images[scenario_id].append(file)
            else:
                 self.scenario_based_slf_test_images[scenario_id] = [file]
        
        '''Compute min and max values for radio map from samples'''
        self.rm_min, self.rm_max = self.PL_min, self.PL_max
    
    @staticmethod
    def _to_log(x):
        if torch.is_tensor(x):
            return torch.clip(10*torch.log10( torch.clip(x + 1e-30, min=1e-30, max=None)), min=-147, max=None)
        else:
            return np.clip(10*np.log10(np.clip(x + 1e-30, a_min=1e-30, a_max=None)),a_min=-147,a_max=None)  # we use 1e-30 here to avoid numerical issues; the output should be stronger than -147 dB, even considering there are small values due to low PSD, i.e., C(k).

    @staticmethod
    def _from_log(x):
        return 10**(x/10.0)
    
    @staticmethod
    def get_scenario_id(file):
        return file.split('/')[-1].split(".png")[0].split("_")[0]
    
    def read_slf(self, file, db_scale=False):
        slf = imread(file)
        slf = (slf/255.0) * (self.PL_max - self.PL_min) + self.PL_min # -147 to -47 dB
        slf = np.clip(slf, a_min=self.PL_thre, a_max=None) # clip the SLF to be no smaller than -127 dB
        if not db_scale:
            slf = self._from_log(slf)
        return slf
    
    def get_original_scale_slf(self, slf, revert_log=True): # input: slf from the diffusion model, normally in [0,1] <-> [-127,-47], same as training process
        if torch.is_tensor(slf):
            slf = torch.clamp(slf, min=0, max=1)
        else:
            slf = np.clip(slf, a_min=0, a_max=1)
        slf = slf*(self.slf_max-self.slf_min)+self.slf_min
        if revert_log:
            slf = self._from_log(slf)
        
        if torch.is_tensor(slf):
            return torch.clamp(slf, min=0, max=None)
        else:
            return np.clip(slf, a_min=0, a_max=None)
    
    def get_original_scale_rm(self, rm, revert_log=True): # input: rm from the diffusion model (ddpm conditioning on the samples), normally in [0,1] <-> [-147,-47], same as training process
        if torch.is_tensor(rm):
            rm = torch.clamp(rm, min=0, max=1)
        else:
            rm = np.clip(rm, a_min=0, a_max=1)
        rm = rm*(self.PL_max-self.PL_min)+self.PL_min
        if revert_log:
            rm = self._from_log(rm)
        
        if torch.is_tensor(rm):
            return torch.clamp(rm, min=0, max=None)
        else:
            return np.clip(rm, a_min=0, a_max=None)

    def get_a_random_sample_with_spec(self, num_emitters, sample_frac, K, train=False):
        if train:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_train_images.keys())) if len(self.scenario_based_slf_train_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_train_images[scenario_id])
        else:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_test_images.keys())) if len(self.scenario_based_slf_test_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_test_images[scenario_id])
        
        print('scenario_id:', scenario_id)
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_file_path = candidate_slf_files[slf_index]
            gt = self.read_slf(load_file_path)
            if self.image_size is not None and self.image_size != gt.shape[0]:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        if K == 1 and num_emitters == 1:
            psd = np.ones((num_emitters,1))
        elif K == 1 and num_emitters > 1:
            # for random PSD values (stored in exp6_mr)
            # psd = np.random.rand(num_emitters,1) * (self.psd_max - self.psd_min) + self.psd_min
            psd = np.ones((num_emitters,1)) * 0.5
        else:
            psd = self.psd_generator.generate_psd(num_emitters).T # (num_emitters, K)

        sc = self.compute_radio_map(slf, psd)
        
        
        '''Load conditioning masks'''
        masks = []
        if self.building_mask_dir:
            building_mask = imread(os.path.join(self.building_mask_dir, f"{scenario_id}.png"))
            building_mask = building_mask/ 255.0
            masks.append(building_mask)
        if self.car_mask_dir:
            car_mask = imread(os.path.join(self.car_mask_dir, f"{scenario_id}.png"))
            car_mask = car_mask/255.0
            masks.append(car_mask)
        if self.road_mask_dir:
            road_mask = imread(os.path.join(self.road_mask_dir, f"{scenario_id}.png"))
            road_mask = road_mask/255.0
            masks.append(road_mask)
        if self.is_sampling_mask:
            sampling_rate = sample_frac
            sampling_mask = np.random.binomial(1, sampling_rate, size=[self.image_size, self.image_size])
            masks.append(sampling_mask)
        if masks:
            masks = np.stack(masks, axis=0)
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(torch.from_numpy(masks)).numpy()
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc, 'mask': masks}
    
    def get_a_random_sample(self, num_emitters, train=False):
        if train:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_train_images.keys())) if len(self.scenario_based_slf_train_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_train_images[scenario_id])
        else:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_test_images.keys())) if len(self.scenario_based_slf_test_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_test_images[scenario_id])
        
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_file_path = candidate_slf_files[slf_index]
            gt = self.read_slf(load_file_path)
            if self.image_size is not None and self.image_size != gt.shape[0]:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        psd = self.psd_generator.generate_psd(num_emitters).T # (num_emitters, K)
        
        sc = self.compute_radio_map(slf, psd)
        
        
        '''Load conditioning masks'''
        masks = []
        if self.building_mask_dir:
            building_mask = imread(os.path.join(self.building_mask_dir, f"{scenario_id}.png"))
            building_mask = building_mask/ 255.0
            masks.append(building_mask)
        if self.car_mask_dir:
            car_mask = imread(os.path.join(self.car_mask_dir, f"{scenario_id}.png"))
            car_mask = car_mask/255.0
            masks.append(car_mask)
        if self.road_mask_dir:
            road_mask = imread(os.path.join(self.road_mask_dir, f"{scenario_id}.png"))
            road_mask = road_mask/255.0
            masks.append(road_mask)
        if self.is_sampling_mask:
            sampling_rate = np.random.uniform(*self.sampling_rate_range)
            sampling_mask = np.random.binomial(1, sampling_rate, size=[self.image_size, self.image_size])
            sampled_gt = gt.squeeze() * sampling_mask
            masks.append(sampled_gt)
        if masks:
            masks = np.stack(masks, axis=0)
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(torch.from_numpy(masks)).numpy()
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc, 'mask': masks}
    
    
    def get_a_random_slf(self, num_emitters, train=False):
        if train:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_train_images.keys())) if len(self.scenario_based_slf_train_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_train_images[scenario_id])
        else:
            scenario_id = np.random.choice(list(id for id in sorted(list(self.scenario_based_slf_test_images.keys())) if len(self.scenario_based_slf_test_images[id]) > num_emitters))
            candidate_slf_files = sorted(self.scenario_based_slf_test_images[scenario_id])
        
        slf_indices = np.random.choice(len(candidate_slf_files), num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_file_path = candidate_slf_files[slf_index]
            gt = self.read_slf(load_file_path)
            if self.image_size is not None and self.image_size != gt.shape[0]:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0) # (num_emitters, H, W)
        psd = np.ones((num_emitters,self.K)) # (num_emitters, K)
        
        sc = self.compute_radio_map(slf, psd)
        
        
        '''Load conditioning masks'''
        masks = []
        if self.building_mask_dir:
            building_mask = imread(os.path.join(self.building_mask_dir, f"{scenario_id}.png"))
            building_mask = building_mask/ 255.0
            masks.append(building_mask)
        if self.car_mask_dir:
            car_mask = imread(os.path.join(self.car_mask_dir, f"{scenario_id}.png"))
            car_mask = car_mask/255.0
            masks.append(car_mask)
        if self.road_mask_dir:
            road_mask = imread(os.path.join(self.road_mask_dir, f"{scenario_id}.png"))
            road_mask = road_mask/255.0
            masks.append(road_mask)
        if self.is_sampling_mask:
            sampling_rate = np.random.uniform(*self.sampling_rate_range)
            sampling_mask = np.random.binomial(1, sampling_rate, size=[self.image_size, self.image_size])
            sampled_gt = gt.squeeze() * sampling_mask
            masks.append(sampled_gt)
        if masks:
            masks = np.stack(masks, axis=0)
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(torch.from_numpy(masks)).numpy()
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc, 'mask': masks}