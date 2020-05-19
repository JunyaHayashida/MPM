import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class RMSE_NormLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        yhat_norm = yhat.pow(2).sum(dim=1, keepdim=True).sqrt()
        y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt()

        return torch.sqrt(self.mse(yhat, y)) + self.mse(yhat_norm, y_norm)


class RMSE_Q_NormLoss(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.mse = nn.MSELoss()
        self.q = q

    def forward(self, yhat, y):
        yhat_norm = yhat.pow(2).sum(dim=1, keepdim=True).sqrt()
        y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt()
        dis = y_norm - yhat_norm
        dis_q = torch.max((self.q - 1) * dis, self.q * dis)
        dis_q_mse = torch.mean((dis_q) ** 2)

        return self.mse(yhat, y) + dis_q_mse
        # return torch.sqrt(self.mse(yhat, y)) + torch.sqrt(dis_q_mse)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
