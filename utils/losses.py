from utils.stattools import normcdf
import torch

def loglikelihood_quantization(x, sample_mask, sigma, b_up, b_low):
    up_diff, low_diff = b_up - x, b_low -x
    up_diff, low_diff = up_diff[sample_mask,:]/sigma, low_diff[sample_mask,:]/sigma
    
    cdf_diff = torch.clip(normcdf(up_diff) - normcdf(low_diff), min=1e-16)
    loss = - torch.sum(torch.log(cdf_diff))
    
    return loss

def loglikelihood_quantization_average(x, sample_mask, sigma, b_up, b_low):
    up_diff, low_diff = b_up - x, b_low -x
    up_diff, low_diff = up_diff[sample_mask,:]/sigma, low_diff[sample_mask,:]/sigma
    
    cdf_diff = torch.clip(normcdf(up_diff) - normcdf(low_diff), min=1e-16)
    loss = - torch.sum(torch.log(cdf_diff))/torch.numel(x)
    
    return loss
    
def loglikelihood_l2distance(x, sample_mask, measurement, sigma):
    
    diff = x[sample_mask,:] - measurement[sample_mask,:]
    loss = torch.sum(diff * diff) / (sigma ** 2 * 2)
    
    return loss
    
def loglikelihood_l2distance_average(x, sample_mask, measurement):
    diff = x[sample_mask,:] - measurement[sample_mask,:]
    loss = torch.mean(diff * diff) / 2 / torch.numel(measurement)
    # loss = torch.mean(diff * diff) / 2 / torch.numel(sample_mask)
    return loss