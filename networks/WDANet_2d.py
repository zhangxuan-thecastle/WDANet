import torch
import torch.nn as nn
from .WEDAN_sys_2d import WDMCADSys2D


def load_pretrained(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    # print(msg)
    del checkpoint
    torch.cuda.empty_cache()


class WDMCAD2D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.agile_former = WDMCADSys2D(**config.MODEL.Params, num_classes=num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.agile_former(x)
        return logits
    
    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        load_pretrained(pretrained_path, self.agile_former)
