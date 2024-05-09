import os
import torch
from config import Config
from baselines.clnet import CLNet
from baselines.ncmnet import NCMNet
from baselines.mgnet import MGNet
from utils.tools import safe_load_weights
from utils.vis_utils import batch_visualization

dataset_list = {'yfcc100m': ['buckingham_palace', 'notre_dame_front_facade', 'reichstag', 'sacre_coeur'],
                'yfcc100m_val': ['united_states_capitol', 'westminster_abbey_1', 'westminster_abbey_2', 'ruins_of_st_pauls', 'piazza_dei_miracoli',
                                 'paris_opera_1', 'lincoln_memorial_statue', 'brandenburg_gate', 'big_ben_2', 'big_ben_1']}


if __name__ == '__main__':
    conf = Config()

    dataset_name = 'yfcc100m'
    scene_list = dataset_list['yfcc100m'][-1:]
    goal = 'only_clnet'

    if goal != 'only_clnet':
        mgnet = MGNet().cuda()
        weight_path = '/home/corrformer/model_zoo/mgnet_yfcc.pth'
        weights_dict = torch.load(os.path.join(weight_path), map_location='cpu')
        safe_load_weights(mgnet, weights_dict['state_dict'])
        mgnet.eval()

        ncmnet = NCMNet().cuda()
        weight_path = '/home/corrformer/model_zoo/ncmnet_yfcc.pth'
        weights_dict = torch.load(os.path.join(weight_path), map_location='cpu')
        safe_load_weights(ncmnet, weights_dict['state_dict'])
        ncmnet.eval()
    else:
        mgnet, ncmnet = None, None

    clnet = CLNet().cuda()
    weight_path = './model_zoo/clnet_yfcc.pth'
    weights_dict = torch.load(os.path.join(weight_path), map_location='cpu')
    safe_load_weights(clnet, weights_dict['state_dict'])
    clnet.eval()

    batch_visualization(mgnet, ncmnet, clnet, scene_list, dataset_name)

    print("##############################################")
    print("Done.")

# CUDA_VISIBLE_DEVICES=5 python prediction_vis.py