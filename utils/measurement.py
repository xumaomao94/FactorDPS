from math import ceil
import torch
from utils.quantization import quantize_uniform
import numpy as np

class SamplingQuantizationWithDithering():
    def __init__(self, device, sample_frac, dithering_type, dithering_sigma, quantization_type, quantization_bit, rm_min_value=None, rm_max_value=None, part_missing=None, hole_ratio=0.5):
        self.sample_frac = sample_frac
        self.dithering_type = dithering_type
        self.quantization_bit = quantization_bit
        self.quantization_type = quantization_type
        self.dithering_sigma = dithering_sigma
        self.device = device
        self.part_missing = part_missing
        self.hole_ratio = hole_ratio
        
        '''Radio Map min and max values for quantization'''
        self.rm_min_value = rm_min_value
        self.rm_max_value = rm_max_value

        '''Setup'''
        self.__setup__()

    def __setup__(self):
        self.__setup__sampling__()
        self.__setup__dithering__()
        self.__setup__quantization__()
    
    def __setup__sampling__(self):
        def sampling_fn(data):
            I,J,K = data.shape
            sample_mask = np.random.choice([False, True], size=(I,J), p=[1-self.sample_frac, self.sample_frac])
            print(f"Subsampling fraction: {self.sample_frac}, Number of samples: {np.sum(sample_mask)}/{I*J}")
            if self.part_missing is not None:
                if self.part_missing == 'half':
                    sample_mask_addition = torch.zeros((I,J), dtype=torch.bool, device=self.device)
                    sample_mask_addition[:, :J//2] = True
                    print(f"Half sampling applied. Number of samples: {torch.sum(sample_mask_addition)}/{I*J}")
                elif self.part_missing == 'quarter':
                    sample_mask_addition = torch.zeros((I,J), dtype=torch.bool, device=self.device)
                    sample_mask_addition[:I//2, :J//2] = True
                    print(f"Quarter sampling applied. Number of samples: {torch.sum(sample_mask_addition)}/{I*J}")
                elif self.part_missing == 'hole':
                    sample_mask_addition = torch.ones((I,J), dtype=torch.bool, device=self.device)
                    hole_size_i = int(I * self.hole_ratio)
                    hole_size_j = int(J * self.hole_ratio)
                    start_i = (I - hole_size_i) // 2
                    start_j = (J - hole_size_j) // 2
                    sample_mask_addition[start_i:start_i+hole_size_i, start_j:start_j+hole_size_j] = False
                    print(f"Hole sampling applied. Number of samples: {torch.sum(sample_mask_addition)}/{I*J}")
                else:
                    raise Exception(f"part_missing type {self.part_missing} not supported!!!")
                sample_mask_addition = sample_mask_addition.cpu().numpy()
                sample_mask = sample_mask & sample_mask_addition
                
            measurement = data.clone()
            measurement[~sample_mask, :] = 0
            return measurement, sample_mask
        self.sampling_fn = sampling_fn

    def __setup__dithering__(self):
        if self.dithering_type == 'gaussian':
            self.dither_fn = lambda x: x + \
                torch.randn_like(x, device=self.device) * \
                self.dithering_sigma
        elif self.dithering_type == 'Inactive':
            self.dither_fn = lambda x: x
        else:
            raise Exception(
                f"Dithering type {self.dithering_type} not supported!!!")

    def __setup__quantization__(self):
        if self.quantization_type == 'uniform':
            self.quantize_fn = lambda x, sample_mask: quantize_uniform(x=x.detach().cpu().numpy(),
                                                        n_bits=self.quantization_bit,
                                                        return_distortion=False,
                                                        original_scale=True,
                                                        min_value = self.rm_min_value,
                                                        max_value=self.rm_max_value,
                                                        sample_mask=sample_mask)[1:4]
        elif self.quantization_type == 'Inactive':
            # b_up = torch.full_like(x, fill_value=-147, device=self.device)
            # self.quantize_fn = lambda x, sample_mask: (x, None, None)
            self.quantize_fn = lambda x, sample_mask: (x, torch.full_like(x, fill_value=-147, device=self.device), torch.full_like(x, fill_value=-147, device=self.device))
        else:
            raise Exception(
                f"Quantization type {self.quantization_type} not supported!!!")

    def __call__(self, data, return_boundaries=False):
        
        '''Sampling'''
        data, sample_mask = self.sampling_fn(data)

        '''Dithering'''
        dithered_x = self.dither_fn(data)

        '''Quantization'''
        quantized_x, b_up, b_low = self.quantize_fn(dithered_x, sample_mask)

        '''Typecast to tensor in device'''
        quantized_x = torch.tensor(quantized_x, device=self.device)
        b_up = torch.tensor(b_up, device=self.device)
        b_low = torch.tensor(b_low, device=self.device)
        
        # Todo: Remove if not needed
        quantized_x[~sample_mask, :] = -147
        b_up[~sample_mask, :] = -47
        b_low[~sample_mask, :] = -147

        if return_boundaries:
            return quantized_x, b_up, b_low, sample_mask
        else:
            return quantized_x, sample_mask