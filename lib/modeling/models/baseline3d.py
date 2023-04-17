import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks, HorizontalPoolingPyramid, BasicConv3d, PackSequenceWrapper


class Baseline3d(BaseModel):
    def build_network(self, model_cfg):
        in_c = model_cfg['channels']

        self.conv3d_block1 = nn.Sequential(
            BasicConv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            BasicConv3d(in_c[1], in_c[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_block2 = nn.Sequential(
            BasicConv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            BasicConv3d(in_c[2], in_c[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.conv3d_block3 = nn.Sequential(
            BasicConv3d(in_c[2], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            BasicConv3d(in_c[3], in_c[3], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL

        sils = ipts[0] # [n, s, c=1, h, w]
        del ipts
        
        n, s, c, h, w = sils.size()
        sils = sils.view(n, 1, s, h, w) # [n, t, s, h, w]
        
        # if s < 3:
        #     repeat = 3 if s == 1 else 2
        #     sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d_block1(sils)
        outs = self.conv3d_block2(outs) 
        outs = self.conv3d_block3(outs) # [n, c, s, h, w]

        outs = self.TP(outs, dim=2, seq_dim=2, seqL=seqL)[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]
        feat = outs.permute(2, 0, 1).contiguous()  # [p, n, c]

        embed_1 = self.FCs(feat)  # [p, n, c]
        embed_2, logits = self.BNNecks(embed_1)  # [p, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed_2 = embed_2.permute(1, 0, 2).contiguous()  # [n, p, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c]

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval
