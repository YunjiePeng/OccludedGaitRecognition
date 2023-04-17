import torch
import copy
import numpy as np
import torch.nn as nn

from utils import np2var, list2var
from data.transform import get_transform
from ..base_model import BaseModel
from ..modules import BasicConv2d, BasicConv3d, HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks, SetBlockWrapper, PackSequenceWrapper, TemporalBlock, SelfSupervisedAlignmentModule

class SpatialAlign(BaseModel):
    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        time_dim = model_cfg['time_dim']

        self.align_module = SelfSupervisedAlignmentModule(**model_cfg['align_cfg'])

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
    
    def init_parameters(self):
        # for m in self.modules():
        for name, m in self.named_modules():
            # print("name:{}, m:{}".format(name, m))
            if name == 'align_module.fc.4':
                continue
                
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def inputs_pretreament(self, inputs):
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        trf_cfgs = self.engine_cfg['transform']
        seq_trfs = get_transform(trf_cfgs)

        requires_grad = bool(self.training)
        if self.training:
            seqs = [np2var(np.asarray([seq_trfs[0](fra) for fra in seqs_batch[0]]), requires_grad=requires_grad).float(),
                    np2var(np.asarray(seqs_batch[1]), requires_grad=False).float()]
        else:
            seqs = [np2var(np.asarray([seq_trfs[0](fra) for fra in seqs_batch[0]]), requires_grad=requires_grad).float()]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL
        
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, 1, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        if self.training:
            occ_labs = ipts[1] # [n, s, 1]
            theta, theta_gt, sils_selfSuper, sils_selfSuperTrans, sils_gt, sils_affine = self.align_module(sils, occ_labs) # [n*s, 2, 3]; [n, s, c, h, w]
        else:
            sils_affine = self.align_module(sils) # [n, s, c, h, w]

        del ipts

        #=====spatial conv=========
        outs = self.spatial_block1(sils_affine.detach()) # [n, s, c, h, w]
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

        if self.training:
            n, s, _, h, w = sils.size()
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed_1, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                    'affine': {'theta': theta, 'theta_gt': theta_gt, 'x': sils_selfSuperTrans, 'x_gt': sils_gt}
                },
                'visual_summary': {
                    'image/4_sils': sils[:, :60].view(60, 1, h, w),
                    'image/4_silsAffine': sils_affine[:, :60].view(60, 1, h, w),
                    'image/1_silsSelfSuper': sils_selfSuper[:, :60],
                    'image/2_silsSelfSuperTrans': sils_selfSuperTrans[:, :60],
                    'image/3_silsGroundTruth': sils_gt[:, :60],
                },
            }
        else:
            retval = {
                'inference_feat': {
                    'embeddings': embed
                }
            }
        return retval
