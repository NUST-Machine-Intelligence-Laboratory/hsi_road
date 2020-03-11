import torch


def _get_activation(activation):
    if activation is None:
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")
    return activation_fn


def iou(pr, gt, eps=1e-7, activation=None):
    activation_fn = _get_activation(activation)
    pr = activation_fn(pr)
    if activation == 'sigmoid':
        pr = (pr > 0.5).float()
        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + eps
        result = (intersection + eps) / union
    else:
        class_num = pr.size()[1]
        pr = torch.argmax(pr, dim=1).view(-1) + 1
        gt = torch.argmax(gt, dim=1).view(-1) + 1
        intersection = pr * (gt == pr).long()  # pr*mask
        area_intersection = torch.histc(intersection, bins=class_num, min=1, max=class_num)
        area_pred = torch.histc(pr, bins=class_num, min=1, max=class_num)
        area_gt = torch.histc(gt, bins=class_num, min=1, max=class_num)
        area_union = area_pred + area_gt - area_intersection
        result = area_intersection.float() / (area_union.float() + eps)
        return result

    return result


class IoUMetric(object):
    __name__ = 'iou'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        self.eps = eps
        self.activation = activation

    def __call__(self, y_pr, y_gt):
        return iou(y_pr, y_gt, eps=self.eps, activation=self.activation)

