import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    __name__ = 'focal_loss'

    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(y_pred, dtype=y_pred.dtype)
        y_pred = torch.clamp(y_pred, 1e-8, 1. - 1e-8)
        loss0 = -self.alpha * (one - y_pred) ** self.gamma * y_target * torch.log(y_pred)
        loss1 = -(1. - self.alpha) * y_pred ** self.gamma * (one - y_target) * torch.log(one - y_pred)
        if self.reduction == 'mean':
            loss = torch.mean(loss0 + loss1)
        elif self.reduction == 'sum':
            loss = torch.sum(loss0 + loss1)
        else:
            loss = loss0 + loss1
        return loss


class CrossEntropy2D(torch.nn.Module):
    __name__ = 'ce'

    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """ logsoftmax+nll
        :param y_logits: NCHW logits
        :param y_target: NCHW one-hot encoded label
        :return: cross_entropy loss
        """

        return F.cross_entropy(y_logits, y_target.argmax(1), reduction=self.reduction, weight=self.weight)


class BinaryCrossEntropy(torch.nn.Module):
    __name__ = 'bec'

    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """ sigmoid+bce
        :param y_logits: NCHW logits
        :param y_target: NCHW one-hot encoded label
        :return: bce
        """
        loss = F.binary_cross_entropy_with_logits(y_logits, y_target, reduction=self.reduction, weight=self.weight)
        return loss


class GHMC(torch.nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label one-hot
        if pred.dim() != target.dim():
            target, label_weight = self._expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

    def _expand_binary_labels(self, labels, label_weights, label_channels):
        ''' make the target as a one-hot label '''
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds] - 1] = 1
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights


def get_loss(loss_name, loss_args):
    loss = None
    if loss_name == 'bce':
        loss = BinaryCrossEntropy(**loss_args)
    elif loss_name == 'ce':
        loss = CrossEntropy2D(**loss_args)
    return loss
