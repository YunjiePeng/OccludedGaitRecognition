import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import BasicConv2d, BasicConv3d, HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks, SetBlockWrapper, PackSequenceWrapper, TemporalBlock


class SpatialTemporalBackbone(BaseModel):
    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        time_dim = model_cfg['time_dim']

        self.spatial_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))
        self.temporal_block1 = TemporalBlock(in_c[1], in_c[1], time_dim)

        self.spatial_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.temporal_block2 = TemporalBlock(in_c[2], in_c[2], time_dim)

        self.spatial_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))
        self.temporal_block3 = TemporalBlock(in_c[3], in_c[3], time_dim)

        self.spatial_block1 = SetBlockWrapper(self.spatial_block1)
        self.spatial_block2 = SetBlockWrapper(self.spatial_block2)
        self.spatial_block3 = SetBlockWrapper(self.spatial_block3)

        self.temporal_block1 = PackSequenceWrapper(self.temporal_block1)
        self.temporal_block2 = PackSequenceWrapper(self.temporal_block2)
        self.temporal_block3 = PackSequenceWrapper(self.temporal_block3)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        del ipts

        #=====spatial conv=========
        outs = self.spatial_block1(sils) # [n, s, c, h, w]
        #=====spatial preserving (sp) temporal conv=========
        outs = outs + self.temporal_block1(outs, seqL).permute(1, 0, 2, 3, 4).contiguous() # [n, s, c, h, w]

        #=====spatial conv=========
        outs = self.spatial_block2(outs) # [n, s, c, h, w]
        #=====spatial preserving (sp) temporal conv=========
        outs = outs + self.temporal_block2(outs, seqL).permute(1, 0, 2, 3, 4).contiguous() # [n, s, c, h, w]
        
        #=====spatial conv=========
        outs = self.spatial_block3(outs) # [n, s, c, h, w]
        #=====spatial preserving (sp) temporal conv=========
        outs = outs + self.temporal_block3(outs, seqL).permute(1, 0, 2, 3, 4).contiguous() # [n, s, c, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, dim=1)[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]

        embed_1 = self.FCs(feat)  # [p, n, c]
        embed_2, logits = self.BNNecks(embed_1)  # [p, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed_2 = embed_2.permute(1, 0, 2).contiguous()  # [n, p, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c]

        embed = embed_1 # for inference

        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
