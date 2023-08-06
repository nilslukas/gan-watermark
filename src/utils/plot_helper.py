import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.utils import normalize_image

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['black', 'yellow', 'red'])
def show_heatmap(img1, img2):
    diff = heatmap(img1, img2)
    #diff = gaussian_filter(diff, sigma)
    print(diff.min(), diff.max())
    cmaps = ["jet", mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['black', 'red'])]

    i = 0
    for cmap in cmaps:
        print(cmap)
        plt.subplot(len(cmaps),3,i+1)
        plt.imshow(normalize_image(img1).permute(1,2,0).numpy(), alpha=1)
        plt.axis("off")
        i+=1
        plt.subplot(len(cmaps),3,i+1)
        plt.imshow(normalize_image(img2).permute(1,2,0).numpy(), alpha=1)
        plt.axis("off")
        i+=1
        plt.subplot(len(cmaps),3,i+1)
        plt.imshow(np.clip(diff, 0, 1) , cmap=cmap, alpha=1, vmin=0., vmax=.3)
        plt.axis('off')
        i+=1
    plt.show()


def heatmap(img1, img2):
    diff = torch.abs(img1 - img2).mean(0)
    #diff = normalize_image(diff)
    return diff.detach().numpy()