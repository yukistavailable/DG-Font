import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

import math

try:
    from models.blocks import FRN, ActFirstResBlk
except BaseException:
    from blocks import FRN, ActFirstResBlk

cd_softmax = torch.nn.Softmax(dim=1)


class Discriminator(nn.Module):
    """Discriminator: (image x, domain y) -> (logit out)."""

    def __init__(
            self,
            image_size=256,
            num_domains=2,
            max_conv_dim=1024,
            input_ch=3, last_kernel=True):
        super(Discriminator, self).__init__()
        """
        num_domains: number of fonts
        """

        dim_in = 64 if image_size < 256 else 32
        blocks = []
        blocks += [nn.Conv2d(input_ch, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(image_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ActFirstResBlk(dim_in, dim_in, downsample=False)]
            blocks += [ActFirstResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]

        if last_kernel:
            last_kernel = int(image_size / 2 ** repeat_num)
        else:
            last_kernel = 4
        blocks += [nn.Conv2d(dim_out, dim_out, last_kernel, 1, 0)]

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

        self.apply(weights_init('kaiming'))

    def forward(self, x, y):
        """
        Inputs:
            - x: images of shape (batch, input_ch, image_size, image_size).
            - y: domain indices of shape (batch).
        Output:
            - out: logits of shape (batch).
            - feat: features of shape (batch, num_domains, 1, 1).
        """
        out = self.main(x)

        # (batch, num_domains*1*1)
        out = out.view(out.size(0), -1)

        # idx = tensor([1,2,3,4,...,y.size(0)-1])
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        softmaxed_out = cd_softmax(out)[idx, y]
        out = out[idx, y]                                         # (batch)
        return out, softmaxed_out

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


if __name__ == '__main__':
    D = Discriminator(64, 10)
    x_in = torch.randn(4, 3, 64, 64)
    y_in = torch.randint(0, 10, size=(4, ))
    out, feat = D(x_in, y_in)
    print(out.shape, feat.shape)
