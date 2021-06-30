import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@HEADS.register_module()
class FCOSHeadOpt(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHeadOpt(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        points = self.get_points(bbox_preds, bbox_preds[0].dtype,
                                 bbox_preds[0].device)
        # TODO(ljm): dig into this func
        with torch.no_grad():
            labels, bbox_targets, centerness_targets, pos_inds, pos_num, points \
                = self.get_targets(points, gt_bboxes, gt_labels)

        # (B*N, x)
        cls_scores, bbox_preds, centernesses = \
                tuple(map(lambda xs: torch.cat(list(map(lambda x: x.reshape(*x.shape[:2], -1) \
                                                                   .permute(0, 2, 1),
                                                        xs)),
                                               dim=1) \
                                          .reshape(-1, xs[0].shape[1]),
                          [cls_scores, bbox_preds, centernesses]))

        bbox_preds, centernesses, points = \
            bbox_preds[pos_inds], centernesses[pos_inds], points[pos_inds]

        decoded_bbox_targets, decoded_bbox_preds = \
            tuple(map(lambda x: torch.cat([pos_points - x[:, :2],
                                           pos_points + x[:, 2:]], dim=1),
                      [bbox_targets, bbox_preds]))

        centerness_denorm = centerness_targets.sum().detach()
        pos_num, centerness_denorm = tuple(map(lambda x, eps: max(reduce_mean(x), eps)), [pos_num, centerness_denorm], [1.0, 1e-6])

        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=pos_num)

        loss_bbox = self.loss_bbox(decoded_bbox_preds, decoded_bbox_targets,
                                   weight=centerness_targets,
                                   avg_factor=centerness_denorm)

        loss_centerness = self.loss_centerness(centernesses, centerness_targets, avg_factor=pos_num)

        return dict(loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        points = self.get_points(bbox_preds, bbox_preds[0].dtype,
                                       bbox_preds[0].device)
        # NOTE(ljm): this is the big batch_size inference version
        #            for small batch_size inference, please use Adela official version
        bboxes, scores, labels, keeps = multi_apply(self.get_bboxes_single_level,
                                                    bbox_preds, cls_scores, centernesses, points)
        bboxes, scores, labels, keeps = tuple(map(partial(torch.cat, dim=1),
                                                  [bboxes, scores, labels, keeps]))
        result_list = list(map(self.get_bboxes_single_image, bboxes, scores, labels, keeps, img_metas))
        return result_list

    def get_bboxes_single_image(self, bbox, score, label, keep, img_meta):
        bbox, score, label = bbox[keep], score[keep], label[keep]
        det, keep = batched_nms(bbox, score, label, self.nms)
        det, keep = det[self.pos_nms_num], keep[self.pos_nms_num]
        label = label[keep]
        # TODO(ljm): move this func into a proper place
        def post_process(bbox, img_meta):
            (h, w), scale_factor = img_meta['img_shape'][:2], img_meta['scale_factor']
            bbox[:, :4] = bbox[:, 4] * scale_factor
            bbox[:, [0, 2]] = bbox[:, [0, 2]].clamp(min=0, max=w)
            bbox[:, [1, 3]] = bbox[:, [1, 3]].clamp(min=0, max=h)
        det = post_process(det, img_meta)
        return det, label

    def get_bboxes_single_level(self, bbox_pred, score, centerness, point):
        bbox_pred, score, centerness = tuple(map(lambda x: x.reshape(*x.shape[:2], -1) \
                                                           .permute(0, 2, 1),
                                                 [bbox_pred, score, centerness]))
        score, centerness = score.sigmoid(), centerness=centerness.sigmoid()
        B, N, C = score.shape
        # TODO(ljm): cache points with batch B
        point = point.unsqueeze(0).expand(B, -1, 2)
        score = score.reshape(B, -1) # (B, N*C)

        keep = score > self.pre_nms_thres
        B_inds, NC_inds = keep.nonzero(as_tuple=True)
        num_scores, score_keep = keep.sum(dim=-1), score[keep]
        score = score.new_zeros(B, max(num_scores))
        pos_inds = NC_inds.new_zeros(B, max(num_scores))
        start = 0
        for idx, num_score in enumerate(num_scores):
            end = start + num_score
            score[idx, :num_score] = score_keep[start:end]
            pos_inds[idx, :num_score] = NC_inds[start:end]
            start = end

        score, score_inds = score.topk(self.pre_nms_num, dim=-1, sorted=False) # (B, K)
        score_inds = pos_inds[:, score_inds]
        pos_inds, label = score_inds // C, score_inds % C

        batch_inds = torch.arange(B).view(-1, 1).expand_as(pos_inds).long()
        bbox_pred = bbox_pred[batch_inds, pos_inds, :]
        centerness = centerness[batch_inds, pos_inds, :]
        point = point[batch_inds, pos_inds, :]

        keep = score > self.pre_nms_thres
        score = (score * centerness).sqrt()
        bbox = torch.cat([point - bbox_pred[..., :2],
                          point + bbox_pred[..., 2:]], dim=-1)
        return bbox, score, label, keep

    def get_points(self, featmaps, dtype, device, flatten=False):
        # TODO(ljm): cache list, instead of return iterator
        return list(map(self._get_points_single, featmaps, self.strides,
                                                 dtype=dtype,
                                                 device=device,
                                                 flatten=flatten))

    def _get_points_single(self,
                           featmap,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        with torch.no_grad():
            h, w = featmap.shape[-2:]
            y_range, x_range = tuple(map(partial(torch.arange, dtype=dtype, device=device), [h, w]))
            y, x = torch.meshgrid(y_range, x_range)
            points = torch.stack([x, y], dim=-1).reshape(-1, 2) * stride + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        num_points, regress_ranges = \
            multi_apply(lambda point, regress_range: point.shape[0],
                                                     point.new_tensor(regress_range) \
                                                          .unsqueeze(0) \
                                                          .expand_as(point),
                        points, self.regress_ranges)
        points, regress_ranges = tuple(map(partial(torch.cat, dim=0),
                                           [points, regress_ranges]))
        num_gts = list(map(lambda x: x.shape[0], gt_labels_list))
        B, max_num_gt = len(num_gts), max(num_gts)

        gt_bboxes, gt_labels = \
            tuple(map(lambda xs: points.new_zeros(B, max_num_gt, *xs[0].shape[1:]),
                      [gt_bboxes_list, gt_labels_list]))

        for idx, num_gt in emunerate(num_gts):
            gt_bboxes[idx, :num_gt] = gt_bboxes_list[idx]
            gt_labels[idx, :num_gt] = gt_labels_list[idx]

        # (B, N, max_num_gt, 1/2/4)
        gt_bboxes, points, gt_labels, regress_ranges = \
            gt_bboxes.unsqueeze(1), points.unsqueeze(1).unsqueeze(0), \
            gt_labels.unsqueeze(1), regress_ranges.unsqueeze(1).unsqueeze(0)

        bbox_targets = torch.cat([points - gt_bboxes[..., :2],
                                 gt_bboxes[..., 2:] - points], dim=-1)

        if self.center_sampling:
            gt_centers = (gt_bboxes[..., 2:] + gt_bboxes[..., :2]) / 2
            stride = gt_centers.new_zeros(1, points.shape[1], 1, 1)
            start = 0
            for idx, num_point in enumerate(num_points):
                end = start + num_point
                stride[:, start:end, 0, 0] = self.strides[idx]
                start = end
            radius = self.center_sample_radius * stride
            center_inner_bbox = torch.cat([torch.max(gt_centers - radius, gt_bboxes[..., :2]),
                                           torch.min(gt_centers + radius, gt_bboxes[..., 2:])], dim=-1)
            center_inner_bbox_target = torch.cat([points - center_inner_bbox[..., :2],
                                                  center_inner_bbox[..., 2:] - points], dim=-1)
            is_in_bbox = center_inner_bbox_target.min(dim=-1)[0] > 0
        else:
            is_in_bbox = bbox_targets.min(dim=-1)[0] > 0

        if self.norm_on_bbox:
            bbox_targets = bbox_targets / strides

        max_bbox_targets = bbox_targets.max(dim=-1)[0]
        is_in_regress_range = ((max_bbox_targets >= regress_ranges[..., 0]) & \
                               (max_bbox_targets <= regress_ranges[..., 1]))

        wh = gt_bboxes[..., 2:] - gt_bboxes[..., :2]
        areas = wh[..., 0] * wh[..., 1]

        # (B, N, max_num_gt)
        areas, gt_labels = tuple(map(lambda x: x.expand_as(is_in_bbox), [areas, gt_labels]))
        areas = areas.contiguous()

        areas[~is_in_bbox or ~is_in_regress_range] = INF

        # (B*N, max_num_gt, 1/4)
        areas, gt_labels, bbox_targets = \
            tuple(map(lambda x: x.reshape(-1, *x.shape[2:]), [areas, gt_labels, bbox_targets]))

        # (B*N)
        min_area, min_area_inds = areas.min(dim=-1)

        batch_inds = torch.arange(B).view(-1, 1).expand_as(pos_inds).long()
        labels = gt_labels[batch_inds, min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[batch_inds, min_area_inds]

        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().squeeze(-1)
        pos_num = bbox_targets.new_tensor(pos_inds.shape[0])

        bbox_targets = bbox_targets[pos_inds]

        left_right, top_bottom = bbox_targets[:, [0, 2]], bbox_targets[:, [1, 3]]
        centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] *
                              top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]).sqrt()

        points = points.expand(B, *points.shape[1:]).reshape(-1, points.shape[-1])

        return labels, bbox_targets, centerness_targets, pos_inds, pos_num, points
