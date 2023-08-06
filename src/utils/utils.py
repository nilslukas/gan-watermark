import os
from os.path import dirname, isdir

import matplotlib
import numpy as np
import torch
import torchvision
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm


def rnd_choice(elems, n):
    a = np.arange(len(elems))
    np.random.shuffle(a)
    return torch.from_numpy(a[:n])

def activate_gradients(wm_gan, args):
    print(f"Sampling params up to depth: {args.param_depth}!")
    wm_gan.activate_gradients()

    ctr, all_params_ctr = 0, 0
    masks, hooks = [], []

    # a hook to modify the gradients with a mask.
    def get_mask_grad(j):
        def mask_grad(grad):
            mask = masks[j]
            return (torch.flatten(grad) * mask).view(*grad.shape)
        return mask_grad

    pbar = tqdm(wm_gan.parameters()[:args.param_depth])
    for i, param in enumerate(pbar):
        flattened_params = torch.flatten(param)
        all_params_ctr += len(flattened_params)

        idx = rnd_choice(flattened_params, n=int(len(flattened_params)))
        ctr += len(idx)

        mask = torch.zeros_like(flattened_params)
        mask[idx] += 1
        masks.append(mask)

        hooks.append(param.register_hook(get_mask_grad(i)))
        pbar.set_description(f"Watermarking {ctr}/{all_params_ctr} ({ctr/all_params_ctr*100:.2f}% parameters!")
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def parse_yaml(yaml_path):
	""" Loads a yaml file """
	with open(yaml_path, 'r') as stream:
		data_loaded = yaml.safe_load(stream)
	return data_loaded

def compute_bitwise_acc(message, msg_pred):
    msg_pred_cpy = msg_pred.clone().detach()
    msg_pred_cpy[msg_pred_cpy > 0.5] = 1.0
    msg_pred_cpy[msg_pred_cpy <= 0.5] = 0.0
    bitwise_acc = (message == msg_pred_cpy).float().mean(dim=1).mean().item()
    return bitwise_acc * 100

def norm(x, x_range):
    """ Norms a data point """
    x_min, x_max = x_range
    if x_min == 0 and x_max == 1:
        return (x + 1) / 2
    elif x_min == -1 and x_max == 1:
        return x * 2 - 1
    else:
        raise NotImplementedError

def save_tensor_as_pdf(x, file_name):
        grid = torchvision.utils.make_grid(x, nrow=1, range=(-1, 1), scale_each=True, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(file_name, format="pdf", bbox_inches='tight')


def normalize_image(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def plot_images(x: torch.Tensor, n_row=2, dpi=400, savefig=None, title=None):
    x = x.detach().cpu()
    x = normalize_image(x)
    matplotlib.rcParams["figure.dpi"] = dpi
    grid_img = torchvision.utils.make_grid(x,nrow=n_row, range=(-1, 1), scale_each=True, normalize=True)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    if savefig is not None and isdir(dirname(savefig)):
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()
    if savefig is not None and isdir(dirname(savefig)):
        first_img = savefig.replace(".png", ".pdf")
        save_tensor_as_pdf(x[[0,3,6]], first_img)
        print(f"Saved first img at '{os.path.abspath(first_img)}' figure at '{os.path.abspath(savefig)}'")
