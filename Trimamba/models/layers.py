import torch
import torch.nn as nn

# Modality Embedding
class ModalityEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()  # 0: fusion, 1: visual, 2: text, 3: audio
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=input_dim)

    def forward(self, x, modality_index):
        batch_size, seq_len, _ = x.shape
        modality_tensor = torch.full((batch_size, seq_len), modality_index, dtype=torch.long, device=x.device)
        return self.embedding(modality_tensor)


class ListMLELoss(nn.Module):
    def __init__(self, top_k=None, temperature=1.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, pred, target, mask=None):
        loss = torch.tensor(0.0, device=pred.device)
        count = 0

        for batch_index in range(pred.size(0)):
            if mask is not None:
                valid = mask[batch_index]
                pred_valid = pred[batch_index][valid]
                target_valid = target[batch_index][valid]
            else:
                pred_valid = pred[batch_index]
                target_valid = target[batch_index]

            if pred_valid.numel() < 2:
                continue

            total_steps = pred_valid.size(0)
            top_k = self.top_k if self.top_k and self.top_k < total_steps else total_steps
            _, indices = target_valid.topk(top_k)
            pred_sorted = pred_valid[indices] / self.temperature
            log_cumsum = torch.logcumsumexp(pred_sorted.flip(0), dim=0).flip(0)
            loss = loss - (pred_sorted - log_cumsum).mean()
            count += 1

        return loss / max(count, 1)


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, listmle_weight=0.0, temperature=1.0, listmle_top_k=None):
        super().__init__()
        self.mse_weight = mse_weight
        self.listmle_weight = listmle_weight
        self.mse = nn.MSELoss()
        self.listmle = ListMLELoss(top_k=listmle_top_k, temperature=temperature) if listmle_weight > 0 else None

    def forward(self, pred, target, mask=None):
        total = torch.tensor(0.0, device=pred.device)
        loss_dict = {}

        if self.mse_weight > 0:
            mse_loss = self.mse(pred[mask], target[mask]) if mask is not None else self.mse(pred, target)
            total = total + self.mse_weight * mse_loss
            loss_dict["mse"] = mse_loss.item()

        if self.listmle is not None:
            listmle_loss = self.listmle(pred, target, mask)
            total = total + self.listmle_weight * listmle_loss
            loss_dict["listmle"] = listmle_loss.item()

        return total, loss_dict
