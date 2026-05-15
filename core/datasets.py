import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
from core import utils
import cv2
from tqdm import tqdm
from torchvision.transforms.v2 import Resize
from datautils.psd_generate import PSDGenerator
from imageio.v3 import imread 
def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

class RMDataset(data.Dataset):
    def __init__(self, slf_dir, num_peaks_per_psd, K, basis_type, metadata_save_dir, num_emitters, image_size=None, train_frac=None, use_log_scale=True):
        
        super(RMDataset, self).__init__()
        
        '''Initialize PSD Generator'''
        self.num_peaks_per_psd = num_peaks_per_psd
        self.K = K
        self.basis_type = basis_type
        self.image_size = image_size
        self.psd_generator = PSDGenerator(num_peaks_per_psd=self.num_peaks_per_psd, K=self.K, basis_type=self.basis_type, seed=100)
        
        self.slf_train_fract = train_frac
        self.slf_dir = slf_dir
        self.metadata_save_dir = metadata_save_dir
        self.use_log_scale = use_log_scale
        self.train_frac = train_frac
        self.num_emitters = num_emitters
        
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
        
        '''Partition into train and test slf sets'''
        if self.train_frac and 0 <= self.train_frac < 1:
            train_images_num = int(len(self.image_files)*self.train_frac)
            print(f"Using {train_images_num}/{len(self.image_files)} files for training")
            self.image_files = np.random.choice(self.image_files, train_images_num, replace=False)
            np.save(os.path.join(self.metadata_save_dir, "slf_train_files.npy"), self.image_files)
            self.image_files_num = len(self.image_files)
        
        '''Compute min and max values for radio map from samples'''
        self.rm_min, self.rm_max = self.get_min_max_radio_map_values()

        print(f"Saving min and max values for RM...")
        np.save(os.path.join(self.metadata_save_dir, "rm_min_max_values.npy"), np.array([self.rm_min, self.rm_max]))
    
    def get_min_max_radio_map_values(self, sample_num=1000):
        '''Get min and max values for radio map'''
        rm_min, rm_max = np.inf, -np.inf
        for _ in tqdm(range(sample_num), desc="Computing min and max values for radio map", total=sample_num):
            sample = self.get_a_random_sample()
            radio_map=sample['RM']
            if self.use_log_scale:
                radio_map = self._to_log(radio_map)
            rm_min = min(radio_map.min(), rm_min)
            rm_max = max(radio_map.max(), rm_max)
            
        print(f"RM Min: {rm_min}, RM Max: {rm_max}")
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
    
    def get_original_scale_slf(self, slf, revert_log=True):
        slf = slf*(self.slf_max-self.slf_min)+self.slf_min
        if revert_log:
            slf = self._from_log(slf)
        
        if torch.is_tensor(slf):
            return torch.clamp(slf, min=0, max=None)
        else:
            return np.clip(slf, a_min=0, a_max=None)
    
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
        
        slf = slf.unsqueeze(dim=-1)
        psd = psd.unsqueeze(dim=1).unsqueeze(dim=2)
        radio_map = torch.sum(slf * psd, dim=0)
        
        if is_slf_numpy and is_psd_numpy:
            radio_map = radio_map.numpy()
        return radio_map
        
    def get_a_random_sample(self):
        candidate_slf_files = self.image_files
            
        slf_indices = np.random.choice(len(candidate_slf_files), self.num_emitters, replace=False)
        slf = []
        for slf_index in slf_indices:
            load_dir = candidate_slf_files[slf_index]
            data = sio.loadmat(load_dir)
            gt = np.array(data['Sc'][...], dtype=np.float32)
            if self.image_size is not None:
                gt = Resize(self.image_size)(torch.from_numpy(gt).unsqueeze(dim=0)).squeeze().numpy()
            slf.append(gt)
        slf = np.stack(slf, axis=0)
        psd = self.psd_generator.generate_psd(self.num_emitters).T
        
        sc = self.compute_radio_map(slf, psd)
        
        return {'SLF': slf, 'PSD': psd, 'RM': sc}
    
    def data_transform(self, input):
        if self.use_log_scale:
            input = self._to_log(input)
        input = (input -self.rm_min)/ (self.rm_max - self.rm_min) 
        return input
    
    def reverse_transform(self, input, reverse_log=False):
        input = input * (self.rm_max - self.rm_min) + self.rm_min
        if reverse_log:
            input = self._from_log(input)
        return input
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        RM = torch.from_numpy(self.get_a_random_sample()['RM'])
        RM = RM.permute(2, 0, 1)
        RM = self.data_transform(RM).float()
        
        '''Resize RM to specified size'''
        if self.image_size is not None:
            RM = Resize(self.image_size)(RM)
            
        return {'RM': RM}
            

class SLFDataset(data.Dataset):
    def __init__(self, image_dir, save_dir, image_size=None, use_log_scale=False, train_frac=None):
        self.all_files = os.listdir(image_dir)        
        self.image_files = []
        self.metadata_save_dir = save_dir
        for i in self.all_files:
            if is_mat_file(i):
                full_path = os.path.join(image_dir, i)
                self.image_files.append(full_path)
        
        if train_frac and 0 <= train_frac < 1:
            train_images_num = int(len(self.image_files)*train_frac)
            print(f"Using {train_images_num}/{len(self.image_files)} files for training")
            self.image_files = np.random.choice(self.image_files, train_images_num, replace=False)
            np.save(os.path.join(self.metadata_save_dir, "slf_train_files.npy"), self.image_files)

        # #DEBUG
        # self.image_files = self.image_files[:1000]

        self.image_size = image_size
        self.use_log_scale = use_log_scale
        self.__setup__()
        
    
    @staticmethod
    def _to_log(x):
        return 10*np.log10(np.clip(x, a_min=0, a_max=None)+1e-10)
    
    @staticmethod
    def _from_log(x):
        return 10**((x)/10.0)-1e-10
    
    def __setup__(self):

        print('Setting up the dataset...')
        self.min_value = np.inf
        self.max_value = -np.inf
        for file in tqdm(self.image_files):
            data = sio.loadmat(file)
            gt = np.array(data['Sc'][...], dtype=np.float32)
            if self.use_log_scale:
                gt = self._to_log(gt)
            self.min_value = min(self.min_value, gt.min())
            self.max_value = max(self.max_value, gt.max())

        print(self.min_value, self.max_value)
        np.save(os.path.join(self.metadata_save_dir, "slf_min_max_values.npy"), np.array([self.min_value, self.max_value]))

    def data_transform(self, input):
        if self.use_log_scale:
            input = self._to_log(input)
        input = (input -self.min_value)/ (self.max_value - self.min_value) 
        return input
    
    def reverse_transform(self, input, reverse_log=False):
        input = input * (self.max_value - self.min_value) + self.min_value
        if reverse_log:
            input = self._from_log(input)
        return input
    
    def get_raw_data(self, index):
        load_dir = self.image_files[index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['Sc'][...], dtype=np.float32)
        return gt
        
    def __getitem__(self, index):   
        load_dir = self.image_files[index]
        data = sio.loadmat(load_dir)
        gt = np.array(data['Sc'][...], dtype=np.float32)
        gt = np.expand_dims(gt, axis=2)
        gt = self.data_transform(gt)

        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        
        '''Resize SLF to specified size'''
        if self.image_size is not None:
            gt = Resize(self.image_size)(gt)

        return {'SLF': gt}

    def __len__(self):
        return len(self.image_files)
    
class RadioMapSeerSLFDataset(data.Dataset):
    def __init__(self, root_dir, save_dir, gain_folder, image_size=None, train_frac=None, building_mask_folder=None, car_mask_folder=None, road_mask_folder=None, sampling_conditional=False):
        self.PL_min, self.PL_thre, self.PL_max = -147.0, -127.0, -47.0
        self.gain_dir = os.path.join(root_dir, "gain", gain_folder )
        all_files = os.listdir(self.gain_dir)        
        self.gain_files = []
        self.metadata_save_dir = save_dir
        self.R_max = 1
        if train_frac is not None:
            self.train_frac = train_frac
        else:
            self.train_frac = 0.8916 # 625 buildings and 50000 SLFs for training
        for i in all_files:
            if is_png_file(i):
                full_path = os.path.join(self.gain_dir, i)
                self.gain_files.append(full_path)
        
        if train_frac and 0 <= train_frac < 1:
            total_images_num = len(self.gain_files)
            # train_images_num = int(len(self.gain_files)*train_frac)
            total_buildings_num = int(total_images_num / 80) # there should be 701 building maps, each with 80 SLF maps
            train_buildings_num = int(total_buildings_num * train_frac)
            train_images_num = train_buildings_num * 80
            self.building_length = train_buildings_num
            self.length = train_images_num
            print(f"Using the first {train_buildings_num}/{total_buildings_num} building maps (e.g., {train_images_num}/{len(self.gain_files)} SLF files), and {self.length} samples for training")
            
            self.gain_files = [os.path.join(self.gain_dir, f"{building_id}_{slf_id}.png") 
                             for building_id in range(train_buildings_num) 
                             for slf_id in range(80)]
            np.save(os.path.join(self.metadata_save_dir, "slf_train_files.npy"), self.gain_files)

        self.image_size = image_size

        self.__setup__()
        
        '''Conditioning Mask Directories'''
        self.building_mask_dir = os.path.join(root_dir, "png", building_mask_folder) if building_mask_folder else None
        self.car_mask_dir = os.path.join(root_dir, "png", car_mask_folder) if car_mask_folder else None
        self.road_mask_dir = os.path.join(root_dir, "png", road_mask_folder) if road_mask_folder else None
        self.is_sampling_mask = sampling_conditional
        if self.is_sampling_mask:
            self.sampling_rate_range = [0.01,0.1]
    
    def read_gain(self, file):
        gain = imread(file)
        gain = (gain/255.0) * (self.PL_max - self.PL_min) + self.PL_min # -147 to -47 dB
        gain = np.clip(gain, a_min=self.PL_thre, a_max=None) # clip to -127 to -47 dB
        return gain
    
    @staticmethod
    def _to_log(x):
        return np.clip(10*np.log10(np.clip(x, a_min=1e-30, a_max=None)),a_min=-147,a_max=None)  # we use 1e-30 here to avoid numerical issues; the output should be stronger than -147 dB, even considering there are small values due to low PSD, i.e., C(k).
    
    @staticmethod
    def _from_log(x):
        return 10**((np.clip(x,a_min=-147,a_max=None))/10.0)

    def __setup__(self):

        print('Setting up the dataset...')
        
        self.min_value = self.PL_thre
        self.max_value = self.PL_max

        print(self.min_value, self.max_value)
        np.save(os.path.join(self.metadata_save_dir, "slf_min_max_values.npy"), np.array([self.min_value, self.max_value]))

    def data_transform(self, input):
        input = (input -self.min_value)/ (self.max_value - self.min_value) # 0-1 <-> -127 to -47 dB
        return input

    def data_transform_analytical(self, input):
        input = (input -self.PL_min)/ (self.max_value - self.PL_min) # 0-1 <-> -147 to -47 dB, 0.2 - 1 <-> -127 to -47 dB
        return input
    
    def reverse_transform(self, input, reverse_log=False):
        input = input * (self.max_value - self.min_value) + self.min_value
        if reverse_log:
            input = self._from_log(input)
        return input
    
    def reverse_transform_analytical(self, input, reverse_log=False):
        input = input * (self.max_value - self.PL_min) + self.PL_min
        if reverse_log:
            input = self._from_log(input)
        return input
    
    def get_raw_data(self, index):
        gain_file = self.gain_files[index]
        gt = imread(gain_file)
        return gt
        
    def __getitem__(self, index):
        file_index = index
        gain_file = self.gain_files[file_index]
        scenario_id = gain_file.split('/')[-1].split(".png")[0].split("_")[0]
        gt = self.read_gain(gain_file)
        gt = np.expand_dims(gt, axis=2)
        gt = self.data_transform(gt)

        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()
        
        '''Resize SLF to specified size'''
        if self.image_size is not None and self.image_size != gt.shape[1]:
            gt = Resize(self.image_size)(gt)
        
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
            masks = torch.from_numpy(masks).float()
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(masks)
            return {'SLF': gt, 'mask': masks}
        else:
            return {'SLF': gt}

    def __len__(self):
        return len(self.gain_files)
    
class RadioMapSeerRMDataset(data.Dataset):
    def __init__(self, root_dir, save_dir, gain_folder, R_max=None, image_size=None, train_frac=None, building_mask_folder=None, car_mask_folder=None, road_mask_folder=None, sampling_conditional=False, fix_C=False):
        self.PL_min, self.PL_thre, self.PL_max = -147.0, -127.0, -47.0
        self.gain_dir = os.path.join(root_dir, "gain", gain_folder )
        all_files = os.listdir(self.gain_dir)        
        self.gain_files = []
        self.metadata_save_dir = save_dir
        self.fix_C = fix_C
        if R_max is not None:
            self.R_max = R_max
        else:
            self.R_max = 4
        if train_frac is not None:
            self.train_frac = train_frac
        else:
            self.train_frac = 0.8916 # corresponding to 625 buildings and 50000 SLFs for training
        for i in all_files:
            if is_png_file(i):
                full_path = os.path.join(self.gain_dir, i)
                self.gain_files.append(full_path)
        
        if train_frac and 0 <= train_frac < 1:
            total_images_num = len(self.gain_files)
            # train_images_num = int(len(self.gain_files)*train_frac)
            total_buildings_num = int(total_images_num / 80) # there should be 701 building maps, each with 80 SLF maps
            train_buildings_num = int(total_buildings_num * train_frac)
            train_images_num = train_buildings_num * 80
            self.building_length = train_buildings_num
            self.length = train_images_num
            print(f"Using the first {train_buildings_num}/{total_buildings_num} building maps (e.g., {train_images_num}/{len(self.gain_files)} SLF files), and {self.length} samples for training")
            
            self.gain_files = [os.path.join(self.gain_dir, f"{building_id}_{slf_id}.png") 
                             for building_id in range(train_buildings_num) 
                             for slf_id in range(80)]
            np.save(os.path.join(self.metadata_save_dir, "slf_train_files.npy"), self.gain_files)

        self.image_size = image_size

        self.__setup__()
        
        '''Conditioning Mask Directories'''
        self.building_mask_dir = os.path.join(root_dir, "png", building_mask_folder) if building_mask_folder else None
        self.car_mask_dir = os.path.join(root_dir, "png", car_mask_folder) if car_mask_folder else None
        self.road_mask_dir = os.path.join(root_dir, "png", road_mask_folder) if road_mask_folder else None
        self.is_sampling_mask = sampling_conditional
        if self.is_sampling_mask:
            self.sampling_rate_range = [0.01,0.1]
    
    def read_gain(self, file):
        gain = imread(file)
        gain = (gain/255.0) * (self.PL_max - self.PL_min) + self.PL_min # -147 to -47 dB
        gain = np.clip(gain, a_min=self.PL_thre, a_max=None) # clip to -127 to -47 dB
        return gain
    
    @staticmethod
    def _to_log(x):
        return np.clip(10*np.log10(np.clip(x, a_min=1e-30, a_max=None)),a_min=-147,a_max=None)  # we use 1e-30 here to avoid numerical issues; the output should be stronger than -147 dB, even considering there are small values due to low PSD, i.e., C(k).
    
    @staticmethod
    def _from_log(x):
        return 10**((np.clip(x,a_min=-147,a_max=None))/10.0)

    def __setup__(self):

        print('Setting up the dataset...')

        self.min_value = self.PL_thre
        self.max_value = self.PL_max

        print(self.min_value, self.max_value)
        np.save(os.path.join(self.metadata_save_dir, "slf_min_max_values.npy"), np.array([self.min_value, self.max_value]))

    def data_transform_for_slf(self, input):
        input = np.clip(input, a_min=self.PL_thre, a_max=None)
        input = (input -self.min_value)/ (self.max_value - self.min_value) # 0-1 <-> -127 to -47 dB
        return input
    
    def data_transform(self, input):
        input = (input -self.PL_min)/ (self.max_value - self.PL_min) # 0-1 <-> -147 to -47 dB, 0.2 - 1 <-> -127 to -47 dB
        return input
    
    def reverse_transform(self, input, reverse_log=False):
        input = input * (self.max_value - self.PL_min) + self.PL_min
        if reverse_log:
            input = self._from_log(input)
        return input
    
    def get_raw_data(self, index):
        gain_file = self.gain_files[index]
        gt = imread(gain_file)
        return gt
        
    def __getitem__(self, index):
        file_index = index
        
        scenario_id = np.random.randint(0, self.building_length)
        R = np.random.randint(0, self.R_max) + 1  # number of emitters, 1 - self.R_max
        # generate C with shape (K,1) and S with shape (R, H, W)
        # gt_K = np.random.uniform(0.01, 0.99, R)
        if self.fix_C:
            gt_K = np.array([0.5] * R)
        else:
            gt_K = np.random.uniform(1e-32, 0.5, R)
        slf_ids = np.random.choice(80, R, replace=False)
        gt_S = []
        for slf_id in slf_ids:
            gain_file = os.path.join(self.gain_dir, f"{scenario_id}_{slf_id}.png")
            slf_data = self.read_gain(gain_file)
            slf_from_log = self._from_log(slf_data) # should be in the range of 1e-12.7 to 1e-4.7
            gt_S.append(slf_from_log)
        gt_S = np.stack(gt_S, axis=0)
        gt_from_log = np.sum(gt_K[:, np.newaxis, np.newaxis] * gt_S, axis=0)
        gt_from_log = np.expand_dims(gt_from_log, axis=2)
        gt = self._to_log(gt_from_log) # should be in the range of -147 to x dB, with x >= -47 dB
        gt = self.data_transform(gt) # 0-x, with x >= 1.0, 0.2-1.0 corresponds to -127 to -47 dB
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()
        
        
        '''Resize SLF to specified size'''
        if self.image_size is not None and self.image_size != gt.shape[1]:
            gt = Resize(self.image_size)(gt)
        
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
            masks = torch.from_numpy(masks).float()
            
            if self.image_size is not None and self.image_size != masks.shape[1]:
                masks = Resize(self.image_size)(masks)
            # return {'SLF': gt, 'mask': masks, 'gt_K': gt_K, 'gt_SLFs': gt_S} # SLF: the RM; gt_SLFs: (R, H, W), in original domain
            return {'SLF': gt, 'mask': masks} # SLF: the RM; gt_SLFs: (R, H, W), in original domain
        else:
            return {'SLF': gt}

    def __len__(self):
        return self.length
    


class PSDDataset(data.Dataset):
    def __init__(self, num_peaks_per_psd, K, basis_type, save_dir, num_samples=10000):
        self.metadata_save_dir = save_dir
        self.num_peaks_per_psd = num_peaks_per_psd
        self.K = K
        self.basis_type = basis_type
        self.num_samples = num_samples
        self.generator = PSDGenerator(num_peaks_per_psd=self.num_peaks_per_psd, K=self.K, basis_type=self.basis_type)

        self.__setup__()

    def __setup__(self):
        self.data = self.generator.generate_psd(self.num_samples).T
        self.min_value = self.data.min()
        self.max_value = self.data.max()
        print(self.min_value, self.max_value)
        np.save(os.path.join(self.metadata_save_dir, "psd_min_max_values.npy"), np.array([self.min_value, self.max_value]))
    
    def data_transform(self, input):
        return (input - self.min_value) / (self.max_value - self.min_value)

    def reverse_transform(self, input):
        return input * (self.max_value - self.min_value) + self.min_value
    
    def __getitem__(self, index):
        data = self.data[index]
        data = self.data_transform(data)
        data = np.float32(data)
        return {'PSD': torch.from_numpy(data).reshape(1,self.K)}
    
    def __len__(self):
        return self.num_samples


