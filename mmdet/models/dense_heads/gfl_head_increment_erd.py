# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (distance2bbox, bbox_overlaps,
                                   bbox2distance)#___
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from .gfl_head import GFLHead
from ..task_modules.samplers import PseudoSampler
from ..utils import (multi_apply,
                     unpack_gt_instances,
                     unmap,images_to_levels)#___


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@MODELS.register_module()
class GFLHeadIncrementERD(GFLHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 4.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Default: dict(type='GN', num_groups=32,
            requires_grad=True).
        loss_qfl (:obj:`ConfigDict` or dict): Config of Quality Focal Loss
            (QFL).
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
             to 'DistancePointBBoxCoder'.
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max}``
            in QFL setting. Defaults to 16.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_dfl: ConfigType = dict(
                     type='DistributionFocalLoss', loss_weight=0.25),
                 loss_ld: ConfigType = dict(
                     type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 reg_max: int = 16,
                 #___
                 loss_ld_vlr=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_im=dict(type='IMLoss', loss_weight=0),
                 imitation_method='gibox',
                 #___
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='gfl_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_ld = MODELS.build(loss_ld)
        #___
        self.imitation_method = imitation_method
        self.loss_im = MODELS.build(loss_im)
        self.loss_ld_vlr = MODELS.build(loss_ld_vlr)
        self.loss_kd = MODELS.build(loss_kd)
        # self.iou_calculator = build_iou_calculator(
        #     dict(type='BboxOverlaps2D'), )  #在ATSSAssigner里
        #___

    def distill_loss_by_image_single(self,
                                     anchors,
                                     new_cls_scores,
                                     new_bbox_preds,
                                     ori_cls_inds,
                                     ori_box_inds,
                                     ori_cls_scores,
                                     ori_bbox_preds,
                                     dist_loss_weight,
                                     ori_num_classes: int, avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # ===========>  distillation classification (only u+2 * sigma) using l2 loss
        new_topk_cls_scores = new_cls_scores.gather(0,
                                                    ori_cls_inds.unsqueeze(-1).expand(-1, new_cls_scores.size(-1)))
        ori_topk_cls_scores = ori_cls_scores.gather(0,
                                                    ori_cls_inds.unsqueeze(-1).expand(-1, ori_cls_scores.size(-1)))

        loss_dist_cls = dist_loss_weight * self.l2_loss(new_topk_cls_scores, ori_topk_cls_scores)

        # ===========>  distillation regression (only u+2 * sigma) using ld loss
        anchor_centers = self.anchor_center(anchors)
        # ori decode bbox, shape (Num,4)
        ori_bbox_preds_tblr = self.integral(ori_bbox_preds)
        decode_bbox_pred = distance2bbox(anchor_centers, ori_bbox_preds_tblr)

        ori_cls_conf = ori_cls_scores.sigmoid()
        cls_conf, ids = ori_cls_conf.max(dim=-1)

        # nms
        nms_cfg = dict(iou_threshold=0.005)  # 0.005

        thr_bboxes, thr_scores, thr_id = decode_bbox_pred[ori_box_inds], cls_conf[ori_box_inds], \
                                         ids[ori_box_inds]
        _, keep = batched_nms(thr_bboxes, thr_scores, thr_id, nms_cfg)

        nms_bbox_preds = new_bbox_preds.gather(
            0, ori_box_inds.unsqueeze(-1).expand(-1, new_bbox_preds.size(-1)))
        new_topk_bbox_preds = nms_bbox_preds.gather(
            0, keep.unsqueeze(-1).expand(-1, nms_bbox_preds.size(-1)))

        nms_ori_topk_bbox_preds = ori_bbox_preds.gather(
            0, ori_box_inds.unsqueeze(-1).expand(-1, ori_bbox_preds.size(-1)))
        ori_topk_bbox_preds = nms_ori_topk_bbox_preds.gather(
            0, keep.unsqueeze(-1).expand(-1, nms_ori_topk_bbox_preds.size(-1)))

        new_topk_bbox_corners = new_topk_bbox_preds.reshape(-1, self.reg_max + 1)
        ori_topk_pred_corners = ori_topk_bbox_preds.reshape(-1, self.reg_max + 1)

        weight_targets = new_cls_scores.reshape(-1, ori_num_classes)[ori_box_inds].detach().sigmoid()
        weight_targets = weight_targets.max(dim=1)[0][keep.reshape(-1)]
        loss_dist_bbox = dist_loss_weight * self.loss_ld(new_topk_bbox_corners, ori_topk_pred_corners,
                                                         weight=weight_targets[:, None].expand(-1, 4).reshape(
                                                             -1), avg_factor=4.0)

        return loss_dist_cls, loss_dist_bbox

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            stride: Tuple[int], ori_num_classes: int, avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score[:, ori_num_classes:].permute(0, 2, 3,
                                                           1).reshape(-1, self.cls_out_channels - ori_num_classes)

        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes - ori_num_classes  # only optimize the novel classes
        labels[labels == self.num_classes] = bg_class_ind  # only optimize the novel classes

        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    @staticmethod
    def l2_loss(pred, target, reduction='mean'):
        r"""Function that takes the mean element-wise square value difference.
        """
        assert target.size() == pred.size()
        loss = (pred - target).pow(2).float()
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss

    def loss_by_feat(self,
                     ori_outs: Tuple[Tensor],
                     new_outs: Tuple[Tensor],
                     ori_topk_cls_inds,  # for distillation
                     ori_topk_cls_scores,  # for distillation
                     ori_topk_bbox_inds,  # for distillation
                     ori_topk_bbox_preds,  # for distillation
                     ori_num_classes,
                     dist_loss_weight,
                     model,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # ****************************** ori loss **********************************
        cls_scores, bbox_preds = new_outs
        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl, \
        avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            ori_num_classes=ori_num_classes,
            avg_factor=avg_factor)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        # ****************************** distill loss **********************************
        anchor_list = torch.cat(anchor_list, dim=1)
        bbox_preds_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds]
        bbox_preds_list = torch.cat(bbox_preds_list, dim=1)

        ori_cls_scores, ori_bbox_preds = ori_outs

        ori_cls_scores_list = [
            ori_cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes)
            for ori_cls_score in ori_cls_scores]
        ori_cls_scores_list = torch.cat(ori_cls_scores_list, dim=1)

        ori_bbox_preds_list = [
            ori_bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for ori_bbox_pred in ori_bbox_preds]
        ori_bbox_preds_list = torch.cat(ori_bbox_preds_list, dim=1)

        new_cls_scores_list = [
            cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes) for cls_score in cls_scores]
        new_cls_scores_list = torch.cat(new_cls_scores_list, dim=1)

        loss_dist_cls, loss_dist_bbox = multi_apply(
            self.distill_loss_by_image_single,
            anchor_list,
            new_cls_scores_list,
            bbox_preds_list,
            ori_topk_cls_inds,
            ori_topk_bbox_inds,
            ori_cls_scores_list,
            ori_bbox_preds_list,
            dist_loss_weight=dist_loss_weight,
            ori_num_classes=ori_num_classes,
            avg_factor=avg_factor)
        

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_dist_cls=loss_dist_cls,
            loss_dist_bbox=loss_dist_bbox)

    # def loss(self, ori_out: Tuple[Tensor], new_out: Tuple[Tensor],batch_data_samples: SampleList) -> dict:
    def loss(self, ori_outs: Tuple[Tensor], new_outs: Tuple[Tensor], batch_data_samples: SampleList,
             topk_cls_inds, topk_cls_scores, topk_bbox_inds, topk_bbox_preds,
             ori_num_classes, dist_loss_weight, model) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = (ori_outs, new_outs, topk_cls_inds, topk_cls_scores, topk_bbox_inds, topk_bbox_preds,
                       ori_num_classes, dist_loss_weight, model) + (
                          batch_gt_instances, batch_img_metas,
                          batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses


#___


    def loss1(self,
              x,
              out_teacher,
              teacher_x,
              batch_inputs,
              batch_data_samples) -> dict:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        batch_gt_bboxes = []
        batch_gt_labels = []
        for gt_instances in batch_gt_instances:
            batch_gt_bboxes.append(gt_instances.bboxes)
            batch_gt_labels.append(gt_instances.labels)
        # batch_gt_bboxes = batch_gt_instances.bboxes
        # batch_gt_labels = batch_gt_instances.labels
        # batch_gt_bboxes = []
        # batch_gt_labels = []
        # for gt_instances in batch_gt_instances:
        #     batch_gt_bboxes.append(gt_instances['bboxes'])
        #     batch_gt_labels.append(gt_instances['labels'])
        if teacher_x is None:
            loss_inputs1 = (x, out_teacher,
                            batch_img_metas,batch_gt_bboxes,
                            batch_gt_labels,batch_gt_instances_ignore)
        else:
            loss_inputs1 = (x, out_teacher, teacher_x,
                            batch_img_metas,batch_gt_bboxes,
                            batch_gt_labels,batch_gt_instances_ignore)
        losses1 = self.loss_add(*loss_inputs1)
        return losses1

    def loss_add(self,
                      x,
                      out_teacher,
                      teacher_x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs1 = outs + (gt_bboxes, out_teacher, x, img_metas)
        else:
            loss_inputs1 = outs + (gt_bboxes, gt_labels, out_teacher, x,
                                  teacher_x, img_metas)
        losses1 = self.loss_addme(*loss_inputs1, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses1
        else:
            ####补充get_bboxes
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses1, proposal_list



    ####报错@force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss_addme(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             soft_teacher,   #
             x,   #
             teacher_x,   #
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        soft_label, soft_target = soft_teacher   #
        cls_reg_targets = self.get_targets1(  # get_target变了
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, assigned_neg_list,
         im_region_list) = cls_reg_targets   #

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, losses_ld, losses_ld_vlr, losses_kd, losses_kd_neg, losses_im,\
            avg_factor = multi_apply(
                self.loss_single1,   # loss_single1变了
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                soft_target,
                soft_label,
                x,
                teacher_x,
                assigned_neg_list,
                im_region_list,
                num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor) + 1e-6
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        
        return dict(
            # loss_cls=losses_cls,
            # loss_bbox=losses_bbox,
            # loss_dfl=losses_dfl,
            loss_ld=losses_ld,
            loss_ld_vlr=losses_ld_vlr,
            loss_kd=losses_kd,
            # loss_kd_neg=losses_kd_neg,
            loss_im=losses_im,
        )



    def get_targets1(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_vlr_region,
         all_im_region) = multi_apply(
             self._get_target_single1,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        #num_total_remain_neg = sum([max(inds.numel(), 1) for inds in assigned_neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        vlr_regions_list = images_to_levels(all_vlr_region, num_level_anchors)
        im_regions_list = images_to_levels(all_im_region, num_level_anchors)
        # sampled anchors of all images

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, vlr_regions_list, im_regions_list)




    def _get_target_single1(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags1(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        assign_result = self.assigner.assign1(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample1(assign_result, anchors,
                                              gt_bboxes)

        vlr_region = self.assigner.get_vlr_region1(anchors,
                                                  num_level_anchors_inside,
                                                  gt_bboxes, gt_bboxes_ignore,
                                                  gt_labels)

        im_region = self.get_im_region1(
            anchors, gt_bboxes, mode=self.imitation_method)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        labels_neg = anchors.new_full((num_valid_anchors, ),
                                      self.num_classes,
                                      dtype=torch.long)

        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            vlr_region = unmap(vlr_region, num_total_anchors, inside_flags)
            im_region = unmap(im_region, num_total_anchors, inside_flags)

            labels_neg = unmap(
                labels_neg,
                num_total_anchors,
                inside_flags,
                fill=self.num_classes)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, vlr_region, im_region)



    # imitation region
    def get_im_region1(self, bboxes, gt_bboxes, mode='fitnet'):
        assert mode in ['gibox', 'finegrained', 'fitnet', 'decouple']
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        bboxes = bboxes[:, :4]
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assigned_fg = (assigned_gt_inds + 0).float()
        # compute iou between all bbox and gt

        iou = self.iou_calculator(bboxes, gt_bboxes, mode='iou')
        fine_grained = torch.nonzero(iou > 0.5 * iou.max(0)[0])
        assigned_fg[fine_grained[:, 0]] = 1
        gt_flag = torch.zeros(bboxes.shape[0])
        anchor_center = self.anchor_center(bboxes)
        for gt_bbox in gt_bboxes:
            in_gt_flag = torch.nonzero(
                (anchor_center[:, 0] > gt_bbox[0])
                & (anchor_center[:, 0] < gt_bbox[2])
                & (anchor_center[:, 1] > gt_bbox[1])
                & (anchor_center[:, 1] < gt_bbox[3]),
                as_tuple=False)
            gt_flag[in_gt_flag] = 1

        if mode == 'finegrained':
            return assigned_fg
        else:
            return gt_flag

    def get_gi_region1(self, soft_label, cls_score, anchors, bbox_pred,
                      soft_targets, stride):

        teacher_score = soft_label.detach().sigmoid()

        student_score = cls_score.detach().sigmoid()  #[num,80]

        anchor_centers = self.anchor_center(anchors) / stride[0]
        sdistribution = self.integral(bbox_pred)
        tdistribution = self.integral(soft_targets)
        sbox = distance2bbox(anchor_centers, sdistribution)  #[num,4]
        tbox = distance2bbox(anchor_centers, tdistribution)

        z = teacher_score - student_score  #difference between teacher score and student score on the whole locations.
        giscore, index = torch.abs(z).max(dim=1)  #GI scores
        k = z >= 0  #who is bigger
        j = torch.take(
            k, index + self.cls_out_channels *
            (torch.arange(student_score.size(0)).cuda()))
        h = j == 0
        gibox = sbox.new_zeros(sbox.shape)
        gibox[j] = tbox[j] + 0
        gibox[h] = sbox[h] + 0  #GI boxes

        idx_out = torch.ops.torchvision.nms(gibox, giscore, 0.3)[:10]
        return idx_out




    def loss_single1(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, soft_targets, soft_label, x,
                    teacher_x, vlr_region, im_region, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        self.ori_num_classes=3
        cls_score = cls_score[:, self.ori_num_classes:].permute(0, 2, 3,
                                                           1).reshape(-1, self.cls_out_channels - self.ori_num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))

        soft_targets = soft_targets.permute(0, 2, 3,
                                            1).reshape(-1,
                                                       4 * (self.reg_max + 1))   #
        # soft_label = soft_label.permute(0, 2, 3,
        #                                 1).reshape(-1, self.cls_out_channels)  #
        soft_label = soft_label.permute(0, 2, 3,
                                        1).reshape(-1, self.ori_num_classes)  #
        teacher_x = teacher_x.permute(0, 2, 3, 1).reshape(-1, 256)  #
        x = x.permute(0, 2, 3, 1).reshape(-1, 256)   #

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        vlr_region = vlr_region.reshape(-1)  #
        im_region = im_region.reshape(-1)  #

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # bg_class_ind = self.num_classes
        bg_class_ind = self.cls_out_channels - self.ori_num_classes  # only optimize the novel classes
        labels[labels == self.cls_out_channels] = bg_class_ind  # only optimize the novel classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        #gt_inds = (labels != bg_class_ind).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        remain_inds = (vlr_region > 0).nonzero().squeeze(1)   #

        #
        if self.imitation_method == 'gibox':
            gi_idx = self.get_gi_region1(soft_label, cls_score, anchors,
                                        bbox_pred, soft_targets, stride)
            gi_teacher = teacher_x[gi_idx]
            gi_student = x[gi_idx]

            loss_im = self.loss_im(gi_student, gi_teacher)
        elif self.imitation_method == 'decouple':
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            ng_inds = (im_region == 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds],
                                       teacher_x[fg_inds]) + 2 * self.loss_im(
                                           x[ng_inds], teacher_x[fg_inds])
            else:
                loss_im = bbox_pred.sum() * 0
        else:
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds], teacher_x[fg_inds])
            else:
                loss_im = bbox_pred.sum() * 0
        #
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners) #
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            # anchor_centers = self.anchor_center(anchors) / stride[0]
            # in_gt = torch.zeros(x.shape[-1], x.shape[-2], device='cuda')
            # for target in pos_decode_bbox_targets:

            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_targets[pos_inds]  #
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)  #

            target_corners = bbox2distance(pos_anchor_centers, #
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)
            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            #
            # ld loss
            loss_ld = self.loss_ld(
                pred_corners,
                soft_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            loss_kd = self.loss_kd(
                cls_score[pos_inds],
                soft_label[pos_inds],
                weight=label_weights[pos_inds],
                avg_factor=pos_inds.shape[0])
            #
        else:
            loss_ld = bbox_pred.sum() * 0  #
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_kd = bbox_pred.sum() * 0  #
            loss_im = bbox_pred.sum() * 0  #
            weight_targets = bbox_pred.new_tensor(0)
        #
        if len(remain_inds) > 0:
            neg_pred_corners = bbox_pred[remain_inds].reshape(
                -1, self.reg_max + 1)
            neg_soft_corners = soft_targets[remain_inds].reshape(
                -1, self.reg_max + 1)

            remain_targets = vlr_region[remain_inds]

            loss_ld_vlr = self.loss_ld_vlr(
                neg_pred_corners,
                neg_soft_corners,
                weight=remain_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=16.0)
            loss_kd_neg = 0 * self.loss_kd(
                cls_score[remain_inds],
                soft_label[remain_inds],
                weight=label_weights[remain_inds],
                avg_factor=remain_inds.shape[0])
        else:
            loss_ld_vlr = bbox_pred.sum() * 0
            loss_kd_neg = bbox_pred.sum() * 0
        #
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_ld, loss_ld_vlr, loss_kd, loss_kd_neg, loss_im, weight_targets.sum(
        )

def anchor_inside_flags1(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags
