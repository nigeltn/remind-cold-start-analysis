import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, model, device, ewc_lambda=2000):
        super().__init__(model, device)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}     
        self.opt_params = {} 

    def _compute_loss(self, imgs, labels, task_id):
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        
        if self.fisher:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fisher:
                    # fisher * (current - old)^2
                    ewc_loss += (self.fisher[name] * (param - self.opt_params[name]).pow(2)).sum()
            loss += (self.ewc_lambda / 2) * ewc_loss
            
        return loss

    def on_task_complete(self, dataloader, task_id):
        print(f"[EWC] Computing Fisher Matrix for Task {task_id}...")
        self.model.eval()
        
        curr_fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.opt_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    curr_fisher[n] += p.grad.detach().pow(2)

        num_samples = len(dataloader.dataset)
        for n in curr_fisher:
            curr_fisher[n] /= num_samples
            
            if n in self.fisher:
                self.fisher[n] += curr_fisher[n]
            else:
                self.fisher[n] = curr_fisher[n]
                
        print("✅ Fisher Matrix Updated.")