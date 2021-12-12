import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        acc = []
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = (1 - label) * torch.pow(euclidean_distance, 2) + \
                           label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        # logic for accuracy
        for loss_idx in range(len(loss_contrastive)):
            if loss_contrastive[loss_idx] < 1.0 and label[loss_idx] == 0:
                acc.append(1)
            elif loss_contrastive[loss_idx] > 1.0 and label[loss_idx] == 1:
                acc.append(1)
            else:
                acc.append(0)

        return torch.mean(loss_contrastive), np.mean(acc)


class PrototypicalLoss(nn.Module):
    """
    Loss class deriving from Module for the prototypical loss function defined below
    """

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def prototypical_loss(input, target, n_support):
    """
    Compute the prototypes by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the prototypes, computes the
    log_probability for each n_query samples for each one of the current
    classes, loss and accuracy are then computed and returned
    :param input: the model output for a batch of samples
    :param target: ground truth for the above batch of samples
    :param n_support: number of samples to keep in account when computing prototypes for each one of the current classes
    :return: loss_val, acc_val
    """

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        """
        get indexes of support classes
        :param c:
        :return:
        """
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    # getting tge support indexes as per classes
    support_idxs = list(map(supp_idxs, classes))

    # calculating the prototype centre point for each classes in the support batch
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    # getting the query indexes
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    # getting the query samples from the batch coming as output from the model
    query_samples = input.to('cpu')[query_idxs]

    # calculating the distance of query from prototypes
    dists = euclidean_dist(query_samples, prototypes)

    # protoype loss formula
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val
