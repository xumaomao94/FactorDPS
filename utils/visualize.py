import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def get_slf_as_grid(images, plot_type="SLF", data_range=(None,None), title=None, row_num=8):
    images = images[np.newaxis, ...] if len(images.shape) == 2 else images
    
    total_image_num = len(images)
    
    if data_range[0] is None:
        data_range[0] = images.min()
    if data_range[1] is None:
        data_range[1] = images.max()

    fig, axs = plt.subplots(int(np.ceil(total_image_num/row_num)), min(row_num, total_image_num), figsize=(20, 4*int(np.ceil(total_image_num/row_num))))
    for image_idx in range(total_image_num):
        image = images[image_idx].squeeze()
        ax = axs if total_image_num == 1 else (axs[image_idx] if total_image_num <= row_num else axs[image_idx//row_num, image_idx%row_num])
        # ax = axs[image_idx//row_num, image_idx%row_num] if total_image_num > row_num else axs[image_idx]
        mappable=ax.imshow(image, cmap='jet', vmin=data_range[0], vmax=data_range[1])
        ax.set_title(f"{plot_type} {image_idx+1}")
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax)
    if title is not None:
        fig.suptitle(title)
    return fig


def get_psd_as_grid(psd_data, plot_type="PSD", data_range=(None,None), title=None, row_num=8):
    psd_data = psd_data[np.newaxis, ...] if len(psd_data.shape) == 1 else psd_data
    total_psd_num = len(psd_data)

    grid_size = (int(np.ceil(total_psd_num/row_num)), min(row_num, total_psd_num))
    fig, axs = plt.subplots(*grid_size, figsize=(20, 3*int(np.ceil(total_psd_num/row_num))))
    for psd_idx in range(total_psd_num):
        psd = psd_data[psd_idx].squeeze()
        ax = axs if total_psd_num == 1 else (axs[psd_idx] if total_psd_num <= row_num else axs[psd_idx//row_num, psd_idx%row_num])
        # ax = axs[psd_idx//row_num, psd_idx%row_num] if total_psd_num > row_num else axs[psd_idx]
        mappable=ax.plot(psd)
        ax.set_ylim(data_range[0], data_range[1])
        ax.set_xlabel("Bands (k)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{plot_type} {psd_idx+1}")
    
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()

    return fig

