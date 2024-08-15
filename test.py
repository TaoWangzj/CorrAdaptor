import torch
from config import Config
import os
from datasets.CorresDataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from baselines.corradaptor import CorrAdaptor
from utils.train_eval_utils import evaluate
from utils.tools import safe_load_weights


if __name__ == '__main__':
    conf = Config()

    conf.use_ransac_auc = False
    conf.use_ransac_map = False
    conf.series_type = 'clnet'

    test_dataset = CorrespondencesDataset(conf.data_te, conf)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             num_workers=10,
                             collate_fn=collate_fn)
    
    model = CorrAdaptor(conf).cuda()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model has {} parameters".format(param_count))

    # download pre-trained models from links in dir weights
    # for fycc
    weights_dict = torch.load(os.path.join('./weights', 'yfcc_best.pth'), map_location="cuda")

    # for sun3d
    # weights_dict = torch.load(os.path.join('./weights', 'sun3d_best.pth'), map_location="cuda")
    
    safe_load_weights(model, weights_dict['state_dict'])

    aucs5, aucs10, aucs20, va_res, precisions, recalls, f_scores = evaluate(model, test_loader, conf, epoch=i)
    va = [aucs5, aucs10, aucs20, va_res[0] * 100, va_res[1] * 100, va_res[2] * 100, va_res[3] * 100, precisions * 100, recalls * 100, f_scores * 100]

    output = ''
    name = ["AUC@5", "AUC@10", "AUC@20", "mAP5", "mAP10", "mAP15", "mAP20", "Precisions", "Recalls", "F_scores"]
    for i, j in enumerate(va):
        output += name[i] + ": " + str(j) + "\n"

    print(output)

# CUDA_VISIBLE_DEVICES=5 python test.py
