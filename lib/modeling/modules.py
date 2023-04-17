import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import clones, is_list_or_tuple
from torchvision.transforms import Resize, InterpolationMode
from timm.models.vision_transformer import Block

class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)

class NewHorizontalPoolingPyramid():
    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        feat_gap_list = []
        feat_gmp_list = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            feat_gap_list.append(z.mean(-1))
            feat_gmp_list.append(z.max(-1)[0])

        feat_gap = torch.cat(feat_gap_list, -1)
        feat_gmp = torch.cat(feat_gmp_list, -1)
        feat = feat_gap + feat_gmp
        return feat, feat_gap, feat_gmp


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w), *args, **kwargs)
        _ = x.size()
        _ = [n, s] + [*_[1:]]
        return x.view(*_)

class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)

# class PackSequenceWrapper(nn.Module):
#     def __init__(self, func):
#         super(PackSequenceWrapper, self).__init__()
#         self.func = func

#     def forward(self, seqs, seqL, seq_dim=1, **kwargs):
#         """
#             In  seqs: [n, s, ...]
#             Out rets: [n, ...]
#         """
#         if seqL is None:
#             return self.func(seqs, **kwargs)
#         seqL = seqL[0].data.cpu().numpy().tolist()
#         start = [0] + np.cumsum(seqL).tolist()[:-1]

#         rets = []
#         for curr_start, curr_seqL in zip(start, seqL):
#             narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
#             # save the memory
#             # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
#             # ret = []
#             # for seq_to_pooling in splited_narrowed_seq:
#             #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
#             #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
#             rets.append(self.func(narrowed_seq, **kwargs))
#         if len(rets) > 0 and is_list_or_tuple(rets[0]):
#             return [torch.cat([ret[j] for ret in rets])
#                     for j in range(len(rets[0]))]
#         return torch.cat(rets)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out


class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_BN1d:
            p, n, c = x.size()
            x = x.transpose(0, 1).contiguous().view(n, -1)  # [n, p*c]
            x = self.bn1d(x)
            x = x.view(n, p, c).permute(1, 0, 2).contiguous()
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.bn1d)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def fix_BN(model):
    for module in model.modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            module.eval()

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super(TemporalBlock, self).__init__()
        self.temporal_block = nn.Sequential(BasicConv3d(in_c, out_c, kernel_size=(time_dim, 1, 1), stride=(1, 1, 1), padding=(time_dim//2, 0, 0)),
                                        nn.LeakyReLU(inplace=True))
    def forward(self, x):
        """
            x  : [1, s, c, h, w]
            ret: [s, 1, c, h, w] 
        """ 
        out = self.temporal_block(x.permute(0, 2, 1, 3, 4).contiguous()).permute(2, 0, 1, 3, 4).contiguous() # [s, n, c, h, w]
        return out

class SelfSupervisedAlignmentModule(nn.Module):
    def __init__(self, feat_c, h_ratio=0.4, w_ratio=0.4, img_h=64, img_w=44):
        super(SelfSupervisedAlignmentModule, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        
        self.conv_net = nn.Sequential(BasicConv2d(feat_c[0], feat_c[1], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(feat_c[1], feat_c[1], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      BasicConv2d(feat_c[1], feat_c[2], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(feat_c[2], feat_c[2], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      BasicConv2d(feat_c[2], feat_c[3], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      BasicConv2d(feat_c[3], feat_c[3], 3, 1, 1),
                                      nn.LeakyReLU(inplace=True))

        featmap_h = img_h // 4
        featmap_w = img_w // 4
        self.fc = nn.Sequential(nn.Linear(feat_c[3]*featmap_h*featmap_w, feat_c[3]*featmap_h),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(feat_c[3]*featmap_h, feat_c[3]),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(feat_c[3], 3)) # predict 3 params: resize_ratio; translation_x; translation_y

        self.fc[4].weight.data.zero_()
        self.fc[4].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0]))

    def forward(self, x, labs=None):
        """
            x  : [n, s, 1, h, w]
            labs: [n, s, 1]
            ret: [2, 3], [n, s, c, h, w] 
        """
        n, s, c, h, w = x.size()
        x = x.view(n*s, c, h, w)

        # forward
        x_feat = self.conv_net(x)
        n_s, _c, _h, _w = x_feat.size()
        theta = self.fc(x_feat.view(-1, _c*_h*_w)).view(-1, 3) # [n*s, 3]

        theta_affine = torch.zeros([n_s, 2, 3]).to(theta.device)
        theta_affine[:, 0, 0] = theta[:, 0]
        theta_affine[:, 0, 2] = theta[:, 1]
        theta_affine[:, 1, 1] = theta[:, 0]
        theta_affine[:, 1, 2] = theta[:, 2]

        grid = F.affine_grid(theta_affine, x.size(), align_corners=True) 
        if grid.dtype == torch.float16:
            x = x.half()
        x_align = F.grid_sample(x, grid, align_corners=True).view(n, s, c, h, w) # mode='bilinear'/'nearest'

        if labs is None:
            return x_align
        else:
            # Self Supervision
            labs = labs.view(n*s)
            x_selfSuper, x_groundTruth, theta_groundTruth = self.self_supervision_construction(x, labs, self.h_ratio, self.w_ratio)
            x_selfSuper_feat = self.conv_net(x_selfSuper.detach()).view(-1, _c*_h*_w)
            theta_selfSuper = self.fc(x_selfSuper_feat).view(-1, 3)

            theta_affine_selfSuper = torch.zeros([theta_selfSuper.size()[0], 2, 3]).to(theta_selfSuper.device)
            theta_affine_selfSuper[:, 0, 0] = theta_selfSuper[:, 0]
            theta_affine_selfSuper[:, 0, 2] = theta_selfSuper[:, 1]
            theta_affine_selfSuper[:, 1, 1] = theta_selfSuper[:, 0]
            theta_affine_selfSuper[:, 1, 2] = theta_selfSuper[:, 2]
            grid_selfSuper = F.affine_grid(theta_affine_selfSuper, x_selfSuper.size(), align_corners=True)
            if grid_selfSuper.dtype == torch.float16:
                x_selfSuperTrans = F.grid_sample(x_selfSuper.half().detach(), grid_selfSuper, align_corners=True) # [n', c, h, w]
            else:
                x_selfSuperTrans = F.grid_sample(x_selfSuper.detach(), grid_selfSuper, align_corners=True) # [n', c, h, w]
            return theta_selfSuper, theta_groundTruth, x_selfSuper, x_selfSuperTrans, x_groundTruth, x_align
    
    def self_supervision_construction(self, x, labs, h_ratio, w_ratio):
        n_s, c, h, w = x.size()
        rand_occ = torch.rand(labs.size()) # <0.5 nonOcc; >=0.5 occ (>=0.75 h_cut, else w_cut)

        rand_h_cut_ratio = torch.rand(labs.size()) * h_ratio
        rand_h_direction = torch.rand(labs.size()) # <0.5 top; >=0.5 bottom.

        rand_w_cut_ratio = torch.rand(labs.size()) * w_ratio
        rand_w_direction = torch.rand(labs.size()) # <0.5 left; >=0.5 right.
        
        x_selfSuper = []
        x_groundTruth = []
        theta_selfSuper = []
        # count = 0
        for _x, _occ, _h_ratio, _h_direction, _w_ratio, _w_direction, _lab in zip(x, rand_occ, rand_h_cut_ratio, rand_h_direction, rand_w_cut_ratio, rand_w_direction, labs):
            if _lab == 1:
                # skip occluded frames
                continue
            
            if (_occ < 0.5) or ((_h_ratio == 0) and (_occ >= 0.75)) or ((_w_ratio == 0) and (0.5 <= _occ < 0.75)):
                # nonOcc
                x_selfSuper.append(_x)
                x_groundTruth.append(_x)
                _x_theta = torch.tensor([1.0, 0.0, 0.0]).to(x.device)
                theta_selfSuper.append(_x_theta)
            elif _occ < 0.75:
                # horizontal Occ
                _w_list = _x.sum(dim=(0,1))
                _w_left = torch.eq((_w_list != 0), 1).to(torch.long).argmax(dim=0)
                _w_right = (_w_list != 0).cumsum(dim=0).argmax(dim=0)
                _delta_w = _w_right - _w_left + 1

                _x_cut_size = math.ceil(_w_ratio * _delta_w)

                _x_cut = torch.zeros_like(_x)
                if _w_direction < 0.5:
                    _x_cut[:, :, _w_left+_x_cut_size:] = _x[:, :, _w_left+_x_cut_size:]
                else:
                    _x_cut[:, :, :_w_right-_x_cut_size] = _x[:, :, :_w_right-_x_cut_size]
                
                x_groundTruth.append(_x_cut)
                
                # Be consistent with the original data preprocessing
                # Get the median of x axis and regard it as the x center of the person
                sum_point = _x_cut.sum()
                sum_column = _x_cut.sum(dim=(0, 1)).cumsum(dim=0)
                for i in range(w):
                    if sum_column[i] > (sum_point / 2):
                        _x_cut_center = i
                        break

                half_w = w // 2
                left = _x_cut_center - half_w
                right = _x_cut_center + half_w
                if left <= 0 or right >= w:
                    left += half_w
                    right += half_w
                    _ = torch.zeros([c, h, half_w]).to(_x_cut.device)
                    _x_cut = torch.cat((_, _x_cut, _), dim=2)
                _x_cut = _x_cut[:, :, left:right]

                x_selfSuper.append(_x_cut)

                move_x = (half_w - _x_cut_center) / half_w
                _x_theta = torch.tensor([1.0, move_x, 0.0]).to(x.device)
                theta_selfSuper.append(_x_theta)
            else:
                # vertical Occ
                _x_cut_size = math.ceil(_h_ratio * h)
                _x_cut = _x[:, _x_cut_size:, :]  if _h_direction < 0.5 else _x[:, :-_x_cut_size, :]

                _x_gt = torch.zeros_like(_x)
                if _h_direction < 0.5:
                    _x_gt[:, _x_cut_size:, :] = _x[:, _x_cut_size:, :]
                else:
                    _x_gt[:, :-_x_cut_size, :] = _x[:, :-_x_cut_size, :]

                x_groundTruth.append(_x_gt)
                
                c, _h, w = _x_cut.size()
                t_w = int(h / _h * w)
                resize_func = Resize([h, t_w], interpolation=InterpolationMode.BICUBIC) #BICUBIC
                _x_cut = resize_func(_x_cut) # [c, h, t_w]

                # Be consistent with the original data preprocessing
                # Get the median of x axis and regard it as the x center of the person
                sum_point = _x_cut.sum()
                sum_column = _x_cut.sum(dim=0).sum(dim=0).cumsum(dim=0)
                for i in range(t_w):
                    if sum_column[i] > (sum_point / 2):
                        _x_cut_center = i
                        break

                half_w = w // 2
                left = _x_cut_center - half_w
                right = _x_cut_center + half_w
                if left <= 0 or right >= t_w:
                    left += half_w
                    right += half_w
                    _ = torch.zeros([c, h, half_w]).to(_x_cut.device)
                    _x_cut = torch.cat((_, _x_cut, _), dim=2)
                _x_cut = _x_cut[:, :, left:right]

                x_selfSuper.append(_x_cut)

                resize_ratio = h / _h
                move_x = (t_w / 2 - _x_cut_center) * (w / t_w) / (half_w * (1 - h_ratio))
                move_y = (_h - h) / _h if _h_direction < 0.5 else (h - _h) / _h
                _x_theta = torch.tensor([resize_ratio, move_x, move_y]).to(x.device)
                theta_selfSuper.append(_x_theta)
        
        x_selfSuper = torch.stack(x_selfSuper, dim=0)
        x_groundTruth = torch.stack(x_groundTruth, dim=0)
        theta_selfSuper = torch.stack(theta_selfSuper, dim=0)
        return x_selfSuper, x_groundTruth, theta_selfSuper

class MaskedPartReconstructionModule(nn.Module):
    def __init__(self, depth, parts, cycle, block_cfg, num_part=32):
        super(MaskedPartReconstructionModule, self).__init__()
        self.depth = depth
        self.parts = parts
        self.cycle = cycle
        self.num_part = 32
        self.decoder_blocks = nn.ModuleList([
            Block(**block_cfg) for _ in range(depth*parts)
        ])

        self.seqSepFC = SeparateFCs(in_channels=128, out_channels=128, parts_num=32)
        self.posSepFC = SeparateFCs(in_channels=128, out_channels=128, parts_num=32)
        self.recSepFC = SeparateFCs(in_channels=128, out_channels=128, parts_num=32)

    def forward(self, x):
        """
            x: [1, s, c, h, w]
        """
        _, s, c, h, w = x.size()
        p = self.num_part

        x_part = x.permute(3, 4, 1, 2, 0).view(p, -1, s, c) # [p, h*w/p, s, c]
        x_part = x_part.mean(1, keepdim=True) + x_part.max(1, keepdim=True)[0] # [p, 1, s, c]
        x_part = x_part.view(p, s, c)

        x_part_detach = x_part.detach()
        # Max Temporal Pooling --- Get the Sequence-level Feature
        x_part_seqFeat = (x_part_detach.mean(1, keepdim=True) + x_part_detach.max(1, keepdim=True)[0]).repeat(1, s, 1) # [p, s, c]
        x_part_seqFeat = self.seqSepFC(x_part_seqFeat)

        x_recon_input = x_part_detach.unsqueeze(1).repeat(1, s, 1, 1).contiguous() # [p, s, s, c]
        x_mask = (torch.eye(s).to(x.device) == 1) # [s, s]
        x_recon_input[:, x_mask, :] = x_part_seqFeat

        x_posEmbed = self.get_sincos_pos_embed_from_time_dim(s, c, x.device).unsqueeze(0).repeat(p, 1, 1) # [p, s, c] fixed sin-cos positional embedding
        x_posEmbed = self.posSepFC(x_posEmbed).unsqueeze(1) # [p, 1, s, c]
        
        x_recon_out = x_recon_input + x_posEmbed # [p, s, s, c]
        part_h = int(self.num_part / self.parts)
        for part_i in range(self.parts):
            for depth_j in range(self.depth):
                x_recon_out[part_i*part_h:(part_i+1)*part_h] = self.decoder_blocks[int(part_i*self.depth + depth_j)](x_recon_out[part_i*part_h:(part_i+1)*part_h].view(-1, s, c)).view(-1, s, s, c) # [s, s, c]

        x_recon = x_recon_out.masked_select(x_mask.unsqueeze(-1)).view(p, s, c)
        x_recon = self.recSepFC(x_recon) # [p, s, c]
        x_part = x_part.unsqueeze(0).permute(2, 0, 3, 1).contiguous() # [s, 1, c, p]
        x_recon = x_recon.unsqueeze(0).permute(2, 0, 3, 1).contiguous() # [s, 1, c, p]
        x_recover = x_part + x_recon.detach()
        
        return x_part.detach(), x_recon, x_recover

    def get_sincos_pos_embed_from_time_dim(self, seq_len, pos_dim, device):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            # this part calculate the position In brackets
            return [position / (math.pi * math.pow(self.cycle, 2 * (hid_j // 2) / pos_dim)) for hid_j in range(pos_dim)]

        sincos_pos_embed = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(seq_len)]).to(device)
        # [:, 0::2] are all even subscripts, is dim_2i
        sincos_pos_embed[:, 0::2] = torch.sin(sincos_pos_embed[:, 0::2])  # dim 2i
        sincos_pos_embed[:, 1::2] = torch.cos(sincos_pos_embed[:, 1::2])  # dim 2i+1
        return sincos_pos_embed