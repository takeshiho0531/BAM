import torch
import torch.nn.functional as F
from model.ASPP import ASPP
from model.PSPNet import OneModel as PSPNet
from torch import nn


def support_MAP(support_image: torch.Tensor, support_annotation: torch.Tensor) -> torch.Tensor:
    """Meta LearnerのMAP(masked average pooling)の部分を行う関数
    support imageとsupport annotationのアダマール積を取って、average poolingを行う流れ。

    Args:
        support_image (torch.Tensor): 3 or 4次元テンソル。(バッチ数)*チャネル数*h*w. チャネル数3
        support_annotation (torch.Tensor): 3 or 4次元テンソル。(バッチ数)*チャネル数*h*w. チャネル数1

    Returns:
        torch.Tensor: Meta LeanerでMAPした後のテンソル
                      3 or 4次元テンソル。(バッチ数)*チャネル数*h*w. チャネル数3
    """

    support_masked = support_image * support_annotation  # support imageとsupport annotationのアダマール積を取った　# アノテーションが当たってる部分だけのrgb成分が残ってる
    dim_h, dim_w = support_masked.shape[-2:][0], support_masked.shape[-2:][1]
    support_annotation_avg_pool2d = F.avg_pool2d(support_annotation, (dim_h, dim_w)) * dim_h * dim_w + 0.0005
    support_mapped = F.avg_pool2d(input=support_masked, kernel_size=support_masked.shape[-2:]) * dim_h * dim_w / support_annotation_avg_pool2d
    return support_mapped


def get_gram_matrix_ori(feature: torch.Tensor) -> torch.Tensor:
    """Ensembleするために、Shared Encoderから取り出した特徴量のGram Matrixを求める関数

    Args:
        feature (torch.Tensor): 4次元テンソル (バッチ数, チャネル数, h, w)

    Returns:
        torch.Tensor: 3次元テンソル (バッチ数, チャネル数, チャネル数)
    """
    b, c, h, w = feature.shape
    feature = feature.reshape(b, c, h * w)
    feature_T = feature.permute(0, 2, 1)
    feature_norm = feature.norm(2, 2, True)  # feature_norm.shape: (5, 3, 641**2) (5-shot case)
    feature_T_norm = feature_T.norm(2, 1, True)  # feature_T_norm.shape: (5, 641**2, 3)
    gram = torch.bmm(feature, feature_T) / (torch.bmm(feature_norm, feature_T_norm) + 1e-7)  # gram.shape: (5,3,3)
    return gram


# 改善ver.
def get_gram_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Ensembleするために、Shared Encoderから取り出した特徴量のGram Matrixを求める関数
    get_gram_matrix_ori()の改善ver.

    Args:
        matrix (torch.Tensor): 4次元テンソル (バッチ数, チャネル数, h, w)

    Returns:
        torch.Tensor: 3次元テンソル (バッチ数, チャネル数, チャネル数)
    """
    b, c, h, w = matrix.shape
    matrix = matrix.reshape(b, c, h * w)
    matrix /= (matrix.norm(2, 2, True) + 1e-7)
    return torch.bmm(matrix, matrix.permute(0, 2, 1))


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.classes = 2

        # Shared Encoder
        PSPNet_ = PSPNet(args)  # base_learner
        weight_path = 'initmodel/PSPNet/coco/resnet50/best.pth'
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_feature_id = "2"  # layer2から抽出するのが最もよいと実験からわかっている。
        fea_dim = 1024 + 512  # if self.vgg: False
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, query_image_preprocessed, support_images_batch_preprocessed, support_annotations_batch_preprocessed):
        query_image_preprocessed_size = query_image_preprocessed.size()
        bs = query_image_preprocessed_size[0]
        h = int((query_image_preprocessed_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((query_image_preprocessed_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(query_image_preprocessed)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # Support Feature
        support_pro_list = []
        final_support_list = []
        mask_list = []
        support_feat_list = []
        for i in range(self.shot):
            mask = (support_annotations_batch_preprocessed[:, i, :, :] == 1).float().unsqueeze(1)  # support_annotations_batch_preprocessed.shape: torch.Size([1, 5, 3, 641, 641])
            mask_list.append(mask)
            with torch.no_grad():
                support_feat_0 = self.layer0(support_images_batch_preprocessed[:, i, :, :, :])
                support_feat_1 = self.layer1(support_feat_0)
                support_feat_2 = self.layer2(support_feat_1)
                support_feat_3 = self.layer3(support_feat_2)
                mask = F.interpolate(mask, size=(support_feat_3.size(2), support_feat_3.size(3)), mode='bilinear', align_corners=True)
                support_feat_4 = self.layer4(support_feat_3 * mask)
                final_support_list.append(support_feat_4)

            support_feat = torch.cat([support_feat_3, support_feat_2], 1)
            support_feat = self.down_supp(support_feat)
            support_pro = support_MAP(support_feat, mask)
            support_pro_list.append(support_pro)
            support_feat_list.append(eval('support_feat_' + self.low_feature_id))

        # K-Shot Reweighting
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_feature_id))  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for support_item in support_feat_list:
            support_gram = get_gram_matrix(support_item)
            gram_diff = que_gram - support_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # Prior Similarity Mask
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_support_feat in enumerate(final_support_list):
            resize_size = tmp_support_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_support_feat_4 = tmp_support_feat * tmp_mask
            q = query_feat_4
            s = tmp_support_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_support = s
            tmp_support = tmp_support.reshape(bsize, ch_sz, -1)
            tmp_support = tmp_support.permute(0, 2, 1)
            tmp_support_norm = torch.norm(tmp_support, 2, 2, True)

            similarity = torch.bmm(tmp_support, tmp_query) / (torch.bmm(tmp_support_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

        # Support Prototype
        support_pro = torch.cat(support_pro_list, 2)  # [bs, 256, shot, 1]
        support_pro = (weight_soft.permute(0, 2, 1, 3) * support_pro).sum(2, True)

        # Tile & Cat
        concat_feat = support_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)   # 256+256+1
        merge_feat = self.init_merge(merge_feat)

        # Base and Meta
        base_out = self.learner_base(query_feat_4)

        # Meta Decoder
        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)   # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:, 0:1, :, :]  # pm^1  # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:, 1:, :, :]  # pm^0                         # [bs, 1, 60, 60]
        base_map = base_out_soft[:, 1:, :, :].sum(1, True)  # pb^f

        est_map = est_val.expand_as(meta_map_fg)  # adjustment factor

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))  # pm^0とψをマージ
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))  # pm^1とψをマージ

        merge_map = torch.cat([meta_map_bg, base_map], 1)  # pb^fと(pm^0とψ)をマージ
        merge_bg = self.cls_merge(merge_map)  # Ensemble                   # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        return final_out
