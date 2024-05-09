import torch
import time
from config import Config
from thop import profile
from baselines.clnet import CLNet

def paras_flops(model, xs, ys):
    flops, params = profile(model, (xs, ys))
    # flops, params = profile(model, (xs, ))
    print('Params: {}M'.format(round(sum(param.numel() for param in model.parameters()) * 1e-6, 2)))
    print('GFlops: {}'.format(round(flops * 1e-9, 2)))


def test_cost(xs, ys, model, iter_num):
    with torch.no_grad():
        #warm up call
        _=model(xs, ys)
        # _ = model(xs)
        torch.cuda.synchronize()
        a=time.time()
        for _ in range(int(iter_num)):
            _=model(xs, ys)
            # _ = model(xs)
        torch.cuda.synchronize()
        b=time.time()
    print('Average time per run(ms): ',(b-a)/iter_num*1e3)
    print('Peak memory(MB): ',torch.cuda.max_memory_allocated()/1e6)


if __name__ == '__main__':
    conf = Config()

    xs = torch.randn(size=(32, 1, 2000, 4)).cuda()
    ys = torch.randn(size=(32, 2000)).cuda()
    model = CLNet().cuda()
    model.eval()

    test_cost(xs, ys, model, 100)
    paras_flops(model, xs, ys)