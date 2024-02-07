import torch

DEFAULT_EPS = 1e-10


def listMLELoss(y_pred, y_true, listmle_temp=1.):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param listmle_temp: temperature param for the softmax distribution (default:1 no temperature)
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    y_pred = y_pred / listmle_temp

    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))

