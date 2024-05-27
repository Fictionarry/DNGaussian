#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import RenderScene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel



import numpy as np
import matplotlib.cm as cm


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)
            
def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized

depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, near):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    makedirs(render_path, exist_ok=True)

    mask_near = None
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
        mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near
        mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
    gaussians.prune_points_inference(mask_near)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
        with torch.no_grad():
            render_pkg = render(view, gaussians, pipeline, background, inference=True)
        rendering = render_pkg["render"]
        depth = (render_pkg['depth'] - render_pkg['depth'].min()) / (render_pkg['depth'].max() - render_pkg['depth'].min()) + 1 * (1 - render_pkg["alpha"])
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(1 - depth, os.path.join(render_path, 'depth_{0:05d}'.format(idx) + ".png"))

        depth_est = depth.squeeze().cpu().numpy()
        depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        depth_est = torch.as_tensor(depth_est).permute(2,0,1)
        torchvision.utils.save_image(depth_est, os.path.join(render_path, 'cdepth_{0:05d}'.format(idx) + ".png"))

    os.system(f"ffmpeg -i " + render_path + f"/%5d.png -q 2 " + model_path + "/out_{}.mp4 -y".format(model_path.split('/')[-1]))
    os.system(f"ffmpeg -i " + render_path + f"/depth_%5d.png -q 2 " + model_path + "/out_depth_{}.mp4 -y".format(model_path.split('/')[-1]))
    os.system(f"ffmpeg -i " + render_path + f"/cdepth_%5d.png -q 2 " + model_path + "/out_cdepth_{}.mp4 -y".format(model_path.split('/')[-1]))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, near):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        (model_params, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
        gaussians.restore(model_params)
        gaussians.neural_renderer.keep_sigma=True

        scene = RenderScene(dataset, gaussians, load_iteration=iteration, spiral=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "render", scene.loaded_iter, scene.getRenderCameras(), gaussians, pipeline, background, near)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--near", default=0, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.near)