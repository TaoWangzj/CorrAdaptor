from config import Config
import numpy as np
import torch
import cv2
import h5py
import numpy
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

conf = Config()


def quaternion_from_matrix(matrix, isprecise=False):

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
        self.num_kp = num_kp

    def run(self, img):
        img = img.astype(np.uint8)
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp])  # N*2

        return kp[:self.num_kp], desc[:self.num_kp]

def computeNN(desc_ii, desc_jj):
    # [2000, 128]
    # desc_ii, desc_jj = torch.from_numpy(desc_ii).cuda(), torch.from_numpy(desc_jj).cuda()
    # TODO: macos
    desc_ii, desc_jj = torch.from_numpy(desc_ii), torch.from_numpy(desc_jj)
    d1 = (desc_ii**2).sum(1) # [2000]
    d2 = (desc_jj**2).sum(1) # [2000]
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt() # [2000, 2000]
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False) # [2000, 2], [2000, 2]
    nnIdx1 = nnIdx1[:,0] # 选出距离最近的index
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False) # 另外一个维度的距离top
    nnIdx2= nnIdx2.squeeze()
    # mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0]).cuda()).cpu().numpy()
    # TODO: macos
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0])).cpu().numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).cpu().numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.cpu().numpy()]
    return idx_sort, ratio_test, mutual_nearest

def norm_kp(cx, cy, fx, fy, kp):
    # New kp
    kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
    return kp

def preprocessing_sift(img1, img2):

    print("=======> Generating initial matching")
    SIFT = ExtractSIFT(num_kp=2000)
    kpts1, desc1 = SIFT.run(img1)
    kpts2, desc2 = SIFT.run(img2)

    idx_sort, ratio_test, mutual_nearest = computeNN(desc1, desc2)  # 计算互为最近邻
    kpts2 = kpts2[idx_sort[1], :]  # 对keypoint2进行排序

    cx1 = (img1.shape[1] - 1.0) * 0.5
    cy1 = (img1.shape[0] - 1.0) * 0.5
    f1 = max(img1.shape[1] - 1.0, img1.shape[0] - 1.0)
    cx2 = (img2.shape[1] - 1.0) * 0.5
    cy2 = (img2.shape[0] - 1.0) * 0.5
    f2 = max(img2.shape[1] - 1.0, img2.shape[0] - 1.0)
    kpts1_n = norm_kp(cx1, cy1, f1, f1, kpts1)
    kpts2_n = norm_kp(cx2, cy2, f2, f2, kpts2)

    return kpts1, kpts2, kpts1_n, kpts2_n

def draw_matching(img1, img2, pt1, pt2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[h1:h1 + h2, :w2] = img2

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    thickness = 1
    num = 0

    for i in range(pt1.shape[0]):
        x1 = int(pt1[i, 0])
        y1 = int(pt1[i, 1])
        x2 = int(pt2[i, 0])
        y2 = int(pt2[i, 1] + h1)

        cv2.line(vis, (x1, y1), (x2, y2), green, int(thickness))

    return vis

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M


def parse_list_file(data_path, list_file):
    fullpath_list = []
    with open(list_file, "r") as img_list:
        while True:
            # read a single line
            tmp = img_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            fullpath_list += [data_path + line2parse.rstrip("\n")]
    return fullpath_list

def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key][:] # H5py改了

    return dict_from_file

def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    try:
        with h5py.File(dump_file_full_name, 'r') as h5file:
            dict_from_file = readh5(h5file)
    except Exception as e:
        print("Error while loading {}".format(dump_file_full_name))
        raise e

    return dict_from_file

def load_geom(geom_file, scale_factor=1.0, flip_R=False):
    # load geometry file
    geom_dict = loadh5(geom_file)
    # Check if principal point is at the center
    K = geom_dict["K"]
    # assert(abs(K[0, 2]) < 1e-3 and abs(K[1, 2]) < 1e-3)
    # Rescale calbration according to previous resizing
    S = np.asarray([[scale_factor, 0, 0],
                    [0, scale_factor, 0],
                    [0, 0, 1]])
    K = np.dot(S, K)
    geom_dict["K"] = K
    # Transpose Rotation Matrix if needed
    if flip_R:
        R = geom_dict["R"].T.copy()
        geom_dict["R"] = R
    # append things to list
    geom_list = []
    geom_info_name_list = ["K", "R", "T", "imsize"]
    for geom_info_name in geom_info_name_list:
        geom_list += [geom_dict[geom_info_name].flatten()]
    # Finally do K_inv since inverting K is tricky with theano
    geom_list += [np.linalg.inv(geom_dict["K"]).flatten()]
    # Get the quaternion from Rotation matrices as well
    q = quaternion_from_matrix(geom_dict["R"])
    geom_list += [q.flatten()]
    # Also add the inverse of the quaternion
    q_inv = q.copy()
    np.negative(q_inv[1:], q_inv[1:])
    geom_list += [q_inv.flatten()]
    # Add to list
    geom = np.concatenate(geom_list)
    return geom

def parse_geom(geom):

    parsed_geom = {}
    parsed_geom["K"] = geom[:9].reshape((3, 3))
    parsed_geom["R"] = geom[9:18].reshape((3, 3))
    parsed_geom["t"] = geom[18:21].reshape((3, 1))
    parsed_geom["img_size"] = geom[21:23].reshape((2,))
    parsed_geom["K_inv"] = geom[23:32].reshape((3, 3))
    parsed_geom["q"] = geom[32:36].reshape([4, 1])
    parsed_geom["q_inv"] = geom[36:40].reshape([4, 1])

    return parsed_geom

def unpack_K(geom):
    img_size, K = geom['img_size'], geom['K']
    w, h = img_size[0], img_size[1]
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5
    cx += K[0, 2]
    cy += K[1, 2]
    # Get focals
    fx = K[0, 0]
    fy = K[1, 1]
    return cx,cy,[fx,fy]

class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    def run(self, img_path):
        img = cv2.imread(img_path)
        cv_kp, desc = self.sift.detectAndCompute(img, None)

        kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4

        return kp, desc

def norm_kp(cx, cy, fx, fy, kp):
    # New kp
    kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
    return kp

def get_episym(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (
        1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
        1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()

def get_input(img1_path, img2_path, image_fullpath_list, geom_fullpath_list):

    ii, jj = image_fullpath_list.index(img1_path), image_fullpath_list.index(img2_path)

    geom_file_i, geom_file_j = geom_fullpath_list[ii], geom_fullpath_list[jj]
    geom_i, geom_j = load_geom(geom_file_i), load_geom(geom_file_j)
    geom_i, geom_j = parse_geom(geom_i), parse_geom(geom_j)
    image_i, image_j = image_fullpath_list[ii], image_fullpath_list[jj]
    detector = ExtractSIFT(2000)
    kp_i, desc_i = detector.run(image_i)
    kp_j, desc_j = detector.run(image_j)
    kp_i = kp_i[:, :2]
    kp_j = kp_j[:, :2]
    idx_sort, ratio_test, mutual_nearest = computeNN(desc_i, desc_j)
    kp1 = kp_i
    kp2 = kp_j[idx_sort[1], :]

    cx1, cy1, f1 = unpack_K(geom_i)
    K1 = [cx1, cy1, f1]
    cx2, cy2, f2 = unpack_K(geom_j)
    K2 = [cx2, cy2, f2]
    x1 = norm_kp(cx1, cy1, f1[0], f1[1], kp_i)
    x2 = norm_kp(cx2, cy2, f2[0], f2[1], kp_j)
    R_i, R_j = geom_i["R"], geom_j["R"]
    dR = np.dot(R_j, R_i.T)
    t_i, t_j = geom_i["t"].reshape([3, 1]), geom_j["t"].reshape([3, 1])
    dt = t_j - np.dot(dR, t_i)
    dtnorm = np.sqrt(np.sum(dt ** 2))
    dt /= dtnorm
    x2 = x2[idx_sort[1], :]
    xs = np.concatenate([x1, x2], axis=1).reshape(1, -1, 4)
    geod_d = get_episym(x1, x2, dR, dt)
    ys = geod_d.reshape(1, -1)

    return kp1, kp2, xs, ys, dR, dt, K1, K2


def unnorm_kpts(cx, cy, fx, fy, kp):
    cx = cx.float().cpu().numpy()
    cy = cy.float().cpu().numpy()
    fx = fx.float().cpu().numpy()
    fy = fy.float().cpu().numpy()
    kp = np.array([fx, fy]).T * kp + np.array([cx, cy]).T
    return kp


def draw_matching_results(img1_path, img2_path, model_name, inlier_pts1, inlier_pts2, outlier_pts1, outlier_pts2, inliers, outliers):
    img1 = cv2.imread(img1_path)
    b, g, r = cv2.split(img1)
    img1 = cv2.merge([r, g, b])
    img2 = cv2.imread(img2_path)
    b, g, r = cv2.split(img2)
    img2 = cv2.merge([r, g, b])
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    vis = np.ones((max(h1, h2), w1 + w2 + 25, 3), np.uint8) * 255
    vis[:h1, :w1] = img1
    vis[:h2, w1 + 25:w1 + w2 + 25] = img2

    fig = plt.figure()
    plt.imshow(vis)
    ax = plt.gca()
    inlier_color = '#04FE05'
    outlier_color = '#F41513'

    if inliers is None:
        tmp1, tmp2 = inlier_pts1, inlier_pts2
        inlier_pts1, inlier_pts2 = outlier_pts1, outlier_pts2
        outlier_pts1, outlier_pts2 = tmp1, tmp2
        inlier_color = '#F41513'
        outlier_color = '#04FE05'

    # inlier
    for i in range(inlier_pts1.shape[0]):
        x1 = int(inlier_pts1[i, 0])
        y1 = int(inlier_pts1[i, 1])
        x2 = int(inlier_pts2[i, 0] + w1 + 25)
        y2 = int(inlier_pts2[i, 1])

        ax.add_artist(plt.Circle((x1, y1), radius=2.5, color=inlier_color))
        ax.add_artist(plt.Circle((x2, y2), radius=2.5, color=inlier_color))
        ax.plot([x1, x2], [y1, y2], c=inlier_color, linestyle='-', linewidth=1.5)

    # outlier
    # for i in range(outlier_pts1.shape[0]):
    #     x1 = int(outlier_pts1[i, 0])
    #     y1 = int(outlier_pts1[i, 1])
    #     x2 = int(outlier_pts2[i, 0] + w1 + 25)
    #     y2 = int(outlier_pts2[i, 1])
    #
    #     ax.add_artist(plt.Circle((x1, y1), radius=2.5, color=outlier_color))
    #     ax.add_artist(plt.Circle((x2, y2), radius=2.5, color=outlier_color))
    #     ax.plot([x1, x2], [y1, y2], c=outlier_color, linestyle='-', linewidth=1.5)

    plt.axis('off')
    fig.set_size_inches((w1 + w2 + 25) / 100, max(h1, h2) / 100)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    # TODO: matplotlib转ndarray
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    img = Image.frombytes("RGBA", (w, h), buf.tostring())
    img = np.asarray(img)
    img = img[:, :, :3][:, :, ::-1]

    if outliers is not None:
        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, (0, 0), (250, 60), (255, 255, 255), -1)  # 注意在 blk的基础上进行绘制；
        # cv2.rectangle(blk, (0, 0), (145, 105), (255, 255, 255), -1)  # 注意在 blk的基础上进行绘制；
        img = cv2.addWeighted(img, 1.0, blk, 0.2, 0)
        # TODO: 将字输出到图片上
        inlier_text = "{}/{}".format(torch.sum(inliers, dim=0), torch.sum(inliers, dim=0) + torch.sum(outliers, dim=0))
        p = round(float(100 * torch.sum(inliers, dim=0) / (torch.sum(inliers, dim=0) + torch.sum(outliers, dim=0))), 2)
        precision_text = "{}".format(p)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(img, 'Inliers: {}'.format(inlier_text), (0, 25), font, 0.8, (0, 0, 0), 1)
        cv2.putText(img, 'Precision: {}%'.format(precision_text), (0, 52), font, 0.8, (0, 0, 0), 1)

    saved_path = '/home/lab/ltf/vis/'
    cv2.imwrite(saved_path + '/{}+{}.png'.format(img1_path.split('/')[-1][:-4], img2_path.split('/')[-1][:-4]), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    plt.close("all")


def batch_visualization(mgnet, ncmnet, clnet, scene_list, dataset_name):
    for scene_name in scene_list:
        data_path = '/home/0_DATA_ROOT/OANet_Dataset/raw_data/' + dataset_name + '/' + scene_name + '/test/'
        img_list_file = data_path + "images.txt"
        geom_list_file = data_path + "calibration.txt"
        image_fullpath_list = parse_list_file(data_path, img_list_file)
        geom_fullpath_list = parse_list_file(data_path, geom_list_file)
        image_list = [i.split('/')[-2] + '/' + i.split('/')[-1] for i in image_fullpath_list]

        length = len(image_list)
        for i in range(length):
            img1 = image_list[i]

            for j in range(i + 1, length):
                img2 = image_list[j]

                # TODO: 挑选高度一致的image pair
                img00 = cv2.imread(data_path + img1)
                img01 = cv2.imread(data_path + img2)
                h1, w1 = img00.shape[:2]
                h2, w2 = img01.shape[:2]
                if h1 != h2:
                    continue

                kp1, kp2, xs, ys, _, _, _, _ = get_input(data_path + img1, data_path + img2, image_fullpath_list, geom_fullpath_list)

                xs = torch.from_numpy(xs).float().cuda()
                ys = torch.from_numpy(ys).float().cuda()

                if mgnet is not None:
                    logits_mgnet, ys_ds_mgnet, e_hat_list_mgnet, y_hat_mgnet, xs_ds_mgnet = mgnet(xs[None], ys)
                    # TODO: 二次剪枝后可视化
                    gt_geod_d_mgnet = ys_ds_mgnet[-1]
                    is_pos_gt_mgnet = (gt_geod_d_mgnet < conf.obj_geod_th).bool().squeeze().cpu()
                    is_neg_gt_mgnet = (gt_geod_d_mgnet >= conf.obj_geod_th).bool().squeeze().cpu()
                    inliers_mgnet = ((logits_mgnet[-1] >= 0).type(is_pos_gt_mgnet.type()) * is_pos_gt_mgnet).squeeze().cpu()  # TP
                    outliers_mgnet = ((logits_mgnet[-1] >= 0).type(is_pos_gt_mgnet.type()) * is_neg_gt_mgnet).squeeze().cpu()  # FP
                    N = xs.shape[1]
                    w0_mgnet = torch.sort(logits_mgnet[1].squeeze(0), dim=-1, descending=True)[1][:int(N * 0.5)].cpu().numpy().astype(np.int32)
                    w1_mgnet = torch.sort(logits_mgnet[3].squeeze(0), dim=-1, descending=True)[1][:int(N * 0.25)].cpu().numpy().astype(np.int32)
                    kpts1_mgnet = kp1[w0_mgnet]
                    kpts2_mgnet = kp2[w0_mgnet]
                    kpts1_mgnet = kpts1_mgnet[w1_mgnet]
                    kpts2_mgnet = kpts2_mgnet[w1_mgnet]
                else:
                    # TODO: initial correspondence
                    gt_geod_d = ys
                    is_pos_gt = (gt_geod_d < conf.obj_geod_th).bool().squeeze().cpu()
                    is_neg_gt = (gt_geod_d >= conf.obj_geod_th).bool().squeeze().cpu()

                if ncmnet is not None:
                    # TODO: 二次剪枝后可视化
                    logits_ncmnet, ys_ds_ncmnet, e_hat_list_ncmnet, y_hat_ncmnet, xs_ds_ncmnet = ncmnet(xs[None], ys)
                    gt_geod_d_ncmnet = ys_ds_ncmnet[-1]
                    is_pos_gt_ncmnet = (gt_geod_d_ncmnet < conf.obj_geod_th).bool().squeeze().cpu()
                    is_neg_gt_ncmnet = (gt_geod_d_ncmnet >= conf.obj_geod_th).bool().squeeze().cpu()
                    inliers_ncmnet = ((logits_ncmnet[-1] >= 0).type(is_pos_gt_ncmnet.type()) * is_pos_gt_ncmnet).squeeze().cpu()  # TP
                    outliers_ncmnet = ((logits_ncmnet[-1] >= 0).type(is_pos_gt_ncmnet.type()) * is_neg_gt_ncmnet).squeeze().cpu()  # FP
                    N = xs.shape[1]
                    w0_ncmnet = torch.sort(logits_ncmnet[1].squeeze(0), dim=-1, descending=True)[1][:int(N * 0.5)].cpu().numpy().astype(np.int32)
                    w1_ncmnet = torch.sort(logits_ncmnet[3].squeeze(0), dim=-1, descending=True)[1][:int(N * 0.25)].cpu().numpy().astype(np.int32)
                    kpts1_ncmnet = kp1[w0_ncmnet]
                    kpts2_ncmnet = kp2[w0_ncmnet]
                    kpts1_ncmnet = kpts1_ncmnet[w1_ncmnet]
                    kpts2_ncmnet = kpts2_ncmnet[w1_ncmnet]

                # TODO: 全尺寸恢复
                # outliers_mgnet = ((y_hat_mgnet < 1e-4).type(is_pos_gt.type()) * is_neg_gt).squeeze().cpu()
                # inliers_mgnet = ((y_hat_mgnet < 1e-4).type(is_pos_gt.type()) * is_pos_gt).squeeze().cpu()
                # outliers_ncmnet = ((y_hat_ncmnet < 1e-4).type(is_pos_gt.type()) * is_neg_gt).squeeze().cpu()
                # inliers_ncmnet = ((y_hat_ncmnet < 1e-4).type(is_pos_gt.type()) * is_pos_gt).squeeze().cpu()
                # outliers_clnet = ((y_hat_clnet < 1e-4).type(is_pos_gt.type()) * is_neg_gt).squeeze().cpu()
                # inliers_clnet = ((y_hat_clnet < 1e-4).type(is_pos_gt.type()) * is_pos_gt).squeeze().cpu()

                logits_clnet, ys_ds_clnet, e_hat_list_clnet, y_hat_clnet, xs_ds_clnet = clnet(xs[None], ys)
                # TODO: 二次剪枝后可视化
                gt_geod_d_clnet = ys_ds_clnet[-1]
                is_pos_gt_clnet = (gt_geod_d_clnet < conf.obj_geod_th).bool().squeeze().cpu()
                is_neg_gt_clnet = (gt_geod_d_clnet >= conf.obj_geod_th).bool().squeeze().cpu()
                inliers_clnet = ((logits_clnet[-1] >= 0).type(is_pos_gt_clnet.type()) * is_pos_gt_clnet).squeeze().cpu()  # TP
                outliers_clnet = ((logits_clnet[-1] >= 0).type(is_pos_gt_clnet.type()) * is_neg_gt_clnet).squeeze().cpu()  # FP
                N = xs.shape[1]
                w0_clnet = torch.sort(logits_clnet[1].squeeze(0), dim=-1, descending=True)[1][:int(N * 0.5)].cpu().numpy().astype(np.int32)
                w1_clnet = torch.sort(logits_clnet[3].squeeze(0), dim=-1, descending=True)[1][:int(N * 0.25)].cpu().numpy().astype(np.int32)
                kpts1_clnet = kp1[w0_clnet]
                kpts2_clnet = kp2[w0_clnet]
                kpts1_clnet = kpts1_clnet[w1_clnet]
                kpts2_clnet = kpts2_clnet[w1_clnet]

                # mgnet and ncmnet is none
                if (inliers_clnet.sum() > (outliers_clnet.sum() + 100)) and (mgnet is None):
                    print("clnet_inlier_num: {}, clnet_outlier_num: {}".format(inliers_clnet.sum(), outliers_clnet.sum()))
                # mgnet and ncmnet is not none
                # if ((outliers_clnet.sum() + 30) < outliers_ncmnet.sum()) and ((outliers_clnet.sum() + 30) < outliers_mgnet.sum()) and (inliers_clnet.sum() > (outliers_clnet.sum() + 100)):
                #     print("clnet_inlier_num: {}, clnet_outlier_num: {}".format(inliers_clnet.sum(), outliers_clnet.sum()))
                else:
                    continue

                if mgnet is not None:
                    draw_matching_results(data_path + img1, data_path + img2, 'mgnet', kpts1_mgnet[inliers_mgnet], kpts2_mgnet[inliers_mgnet], kpts1_mgnet[outliers_mgnet], kpts2_mgnet[outliers_mgnet], inliers_mgnet, outliers_mgnet)
                    draw_matching_results(data_path + img1, data_path + img2, 'clnet', kpts1_clnet[inliers_clnet],
                                          kpts2_clnet[inliers_clnet], kpts1_clnet[outliers_clnet],
                                          kpts2_clnet[outliers_clnet], inliers_clnet, outliers_clnet)
                else:
                    draw_matching_results(data_path + img1, data_path + img2, 'initial_corrs', kp1[is_pos_gt], kp2[is_pos_gt], kp1[is_neg_gt], kp2[is_neg_gt], None, None)
                    draw_matching_results(data_path + img1, data_path + img2, 'clnet', kpts1_clnet[inliers_clnet],
                                          kpts2_clnet[inliers_clnet], kpts1_clnet[outliers_clnet],
                                          kpts2_clnet[outliers_clnet], inliers_clnet, None)

                if ncmnet is not None:
                    draw_matching_results(data_path + img1, data_path + img2, 'ncmnet', kpts1_ncmnet[inliers_ncmnet], kpts2_ncmnet[inliers_ncmnet], kpts1_ncmnet[outliers_ncmnet], kpts2_ncmnet[outliers_ncmnet], inliers_ncmnet, outliers_ncmnet)