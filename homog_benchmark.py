from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
from config import Config
from utils.tools import safe_load_weights
from baselines.corradaptor import CorrAdaptor


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii).cuda(), torch.from_numpy(desc_jj).cuda()
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0]).cuda()).cpu().numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).cpu().numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.cpu().numpy()]
    return idx_sort, ratio_test, mutual_nearest


class HpatchesHomogBenchmark:
    """Hpatches grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path='/data/sets/liuyicheng/datasets/HPatches') -> None:
        seqs_dir = "hpatches-sequences-release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # Ignore seqs is same as LoFTR.
        self.ignore_seqs = set(["i_contruction", "i_crownnight", "i_dc", "i_pencils", "i_whitebuilding", "v_artisans", "v_astronautis", "v_talent"])
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=1e-5)

    def pre_processing_sift(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path)
        cv_kp1, desc1 = self.sift.detectAndCompute(img1, None)
        kp1 = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp1])
        img2 = cv2.imread(img2_path)
        cv_kp2, desc2 = self.sift.detectAndCompute(img2, None)
        kp2 = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp2])

        idx_sort, ratio_test, mutual_nearest = computeNN(desc1, desc2)

        kp2 = kp2[idx_sort[1], :]
        xs_nonorm = np.concatenate([kp1, kp2], axis=1).reshape(1, -1, 4)

        return xs_nonorm

    def benchmark(self, model):
        homog_dists = []
        for seq_idx, seq_name in tqdm(enumerate(self.seq_names), total=len(self.seq_names)):
            if seq_name in self.ignore_seqs:
                continue
            im1_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im1 = Image.open(im1_path)
            w1, h1 = im1.size
            for im_idx in range(2, 7):
                im2_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im2 = Image.open(im2_path)
                w2, h2 = im2.size
                H = np.loadtxt(os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx)))

                xs = self.pre_processing_sift(im1_path, im2_path)
                xs_nonorm = xs.copy()

                # keypoints normalization is same as DKM.
                xs[:, :, 0] = 2 * (xs[:, :, 0] + 0.5) / w1 - 1
                xs[:, :, 1] = 2 * (xs[:, :, 1] + 0.5) / h1 - 1
                xs[:, :, 2] = 2 * (xs[:, :, 2] + 0.5) / w2 - 1
                xs[:, :, 3] = 2 * (xs[:, :, 3] + 0.5) / h2 - 1

                xs = torch.from_numpy(xs).cuda().unsqueeze(dim=0).float()
                pesudo_ys = xs[:, 0, :, 0]

                res_logits, ys_ds, res_e_hat, y_hat, xs_ds = model(xs, pesudo_ys)
                # res_logits, res_e_hat, y_hat = model(xs)  # oanet series

                mask = y_hat.squeeze().cpu().detach().numpy() < 3e-5

                mkpts0 = xs_nonorm[0, :, :2]
                mkpts1 = xs_nonorm[0, :, 2:]

                mask_kp0 = mkpts0[mask]
                mask_kp1 = mkpts1[mask]
                pos_a, pos_b = mask_kp0, mask_kp1
                try:
                    H_pred, inliers = cv2.findHomography(
                        pos_a,
                        pos_b,
                        method = cv2.RANSAC, # cv2.USAC_MAGSAC and cv2.RANSAC
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0
                corners = np.array([[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]])
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (real_warped_corners[:, :2] / real_warped_corners[:, 2:])
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)

        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        auc = pose_auc(np.array(homog_dists), thresholds)
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }


if __name__ == '__main__':
    conf = Config()
    checkpoint = "/data/sets/liuyicheng/codes/corrformer/model_zoo/yfcc_best.pth"

    model = CorrAdaptor(conf).cuda()
    weights_dict = torch.load(checkpoint, map_location="cuda:0")
    safe_load_weights(model, weights_dict['state_dict'])
    model.eval()

    homog_benchmark = HpatchesHomogBenchmark()
    homog_results = homog_benchmark.benchmark(model)

    print("hpatches_homog_auc_3: {}".format(homog_results['hpatches_homog_auc_3'] * 100))
    print("hpatches_homog_auc_5: {}".format(homog_results['hpatches_homog_auc_5'] * 100))
    print("hpatches_homog_auc_10: {}".format(homog_results['hpatches_homog_auc_10'] * 100))

# CUDA_VISIBLE_DEVICES=5 python homog_benchmark.py