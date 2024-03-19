# A tool to quickly count the mean metrics of one dir
# usage:
# $ python metrics_count.py output/ 6000

import os
import json

import numpy as np 
import sys

dataset_path = sys.argv[1]
model_id = "ours_" + sys.argv[2]


ssims_sk = []
ssims_gs = []
psnrs = []
lpipss = []
avgs = []


def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

def compute_avg_error(psnr, ssim, lpips):
  """The 'average' error used in the paper."""
  mse = psnr_to_mse(psnr)
  dssim = np.sqrt(1 - ssim)
  return np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))


for fname in os.listdir(dataset_path):

    with open(os.path.join(dataset_path, fname, 'results_eval.json')) as f:
        result=json.load(f)
    ssims_sk.append(result[model_id]["SSIM_sk"])
    ssims_gs.append(result[model_id]["SSIM"])
    psnrs.append(result[model_id]["PSNR"])
    lpipss.append(result[model_id]["LPIPS"])
    avgs.append(compute_avg_error(psnrs[-1], ssims_sk[-1], lpipss[-1]))

# print(np.mean(psnrs), np.mean(lpipss), np.mean(ssims_sk), np.mean(ssims_gs), np.mean(avgs))
print(np.mean(psnrs), np.mean(lpipss), np.mean(ssims_sk), np.mean(avgs))