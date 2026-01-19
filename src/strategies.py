import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.buffer import REMINDBufer


class BaseStrategy:
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, task_id=0):
        self.model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            loss = self._compute_loss(imgs, labels, task_id)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def _compute_loss(self, imgs, labels, task_id):
        logits = self.model(imgs)
        return self.criterion(logits, labels)

    def on_task_complete(self, dataloader, task_id):
        pass


class EWCStrategy(BaseStrategy):
    def __init__(self, model, device, ewc_lambda=5000, fisher_sample_size=1024):
        super().__init__(model, device)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.opt_params = {}
        self.fisher_sample_size = fisher_sample_size

    def _compute_loss(self, imgs, labels, task_id):
        unique_classes = torch.unique(labels).sort()[0]

        logits = self.model(imgs)

        masked_logits = logits[:, unique_classes]

        mapped_labels = torch.searchsorted(unique_classes, labels)

        loss_main = self.criterion(masked_logits, mapped_labels)

        loss_ewc = torch.tensor(0.0).to(self.device)

        if self.fisher:
            ewc_sum = 0
            for name, param in self.model.named_parameters():
                if name in self.fisher:
                    fisher_val = self.fisher[name]
                    theta_diff = (param - self.opt_params[name]).pow(2)
                    ewc_sum += (fisher_val * theta_diff).sum()

            loss_ewc = (self.ewc_lambda / 2) * ewc_sum

        return loss_main + loss_ewc

    def on_task_complete(self, dataloader, task_id):
        print(f"\n[EWC] Computing Fisher Matrix for Task {task_id}...")
        self.model.eval()
        self.opt_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        curr_fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        all_labels = []
        for _, y in dataloader:
            all_labels.append(y)
            if len(all_labels) * dataloader.batch_size > 2000:
                break
        unique_classes = torch.unique(torch.cat(all_labels)).sort()[0].to(self.device)

        computed_samples = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            if computed_samples >= self.fisher_sample_size:
                break

            self.model.zero_grad()
            logits = self.model(imgs)
            masked_logits = logits[:, unique_classes]
            mapped_labels = torch.searchsorted(unique_classes, labels)
            loss = F.cross_entropy(masked_logits, mapped_labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    curr_fisher[n] += p.grad.detach().pow(2)
            computed_samples += imgs.size(0)

        for n in curr_fisher:
            if n in self.fisher:
                self.fisher[n] += curr_fisher[n]
            else:
                self.fisher[n] = curr_fisher[n]
        print(f"[EWC] Fisher Matrix Updated.")


class REMINDStrategy(BaseStrategy):
    def __init__(
        self,
        model,
        device,
        buffer_size=100000,
        pq_subspaces=32,
        pq_centroids=256,
        lr=0.001,
    ):
        super().__init__(model, device, lr)

        # dummy feature size initialization
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            dummy_feat = model.forward_G(dummy_input)

        self.feat_shape = dummy_feat.shape[1:]
        self.feature_dim = dummy_feat.numel() // dummy_feat.shape[0]

        self.buffer = REMINDBufer(
            buffer_size=buffer_size,
            feature_dim=self.feature_dim,
            pq_subspaces=pq_subspaces,
            pq_centroids=pq_centroids,
            device=device,
        )

    def train_epoch(self, dataloader, task_id=0):
        self.model.train()

        if task_id > 0:
            self.model.G.eval()
            for param in self.model.G.parameters():
                param.requires_grad = False
        else:
            self.model.G.train()

        total_loss = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if task_id == 0:
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            else:
                with torch.no_grad():
                    z_current = self.model.forward_G(imgs)

                logits_real = self.model.forward_F(z_current)
                loss_real = self.criterion(logits_real, labels)

                z_replay_flat, y_replay = self.buffer.get_batch(len(imgs))
                if z_replay_flat is not None:
                    # Reshape flattened features back to (B, C, H, W)
                    z_replay = z_replay_flat.view(-1, *self.feat_shape)

                    # Pass reconstructed features through plastic F
                    logits_replay = self.model.forward_F(z_replay)
                    loss_replay = self.criterion(logits_replay, y_replay)

                    # Combine losses
                    loss = loss_real + loss_replay
                else:
                    loss = loss_real

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def on_task_complete(self, dataloader, task_id):
        print(f"[REMIND] Archiving memories for Task {task_id}...")

        self.model.eval()
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(self.device)

                z = self.model.forward_G(imgs)
                # Flatten: [B, C, H, W] -> [B, FlatDim]
                z_flat = z.view(z.size(0), -1)

                all_feats.append(z_flat.cpu())
                all_labels.append(labels.cpu())

        all_feats = torch.cat(all_feats)
        all_labels = torch.cat(all_labels)

        if not self.buffer.is_trained:
            self.buffer.train_quantizer(all_feats)

        self.buffer.add_data(all_feats, all_labels)
        print(
            f"[REMIND] Added {len(all_feats)} items. Total Buffer Size: {self.buffer.current_size}"
        )
