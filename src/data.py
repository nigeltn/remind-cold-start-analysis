import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class SplitCIFAR10:
    def __init__(self, root="./data", batch_size=64, pretrained=False, debug=False):
        self.batch_size = batch_size
        self.debug = debug

        if pretrained:
            print(f"[Data] Mode: PRE-TRAINED (Resize 224x224 | ImageNet Stats)")
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet Mean
                        std=[0.229, 0.224, 0.225],  # ImageNet Std
                    ),
                ]
            )
        else:
            print(f"[Data] Mode: SCRATCH (Original 32x32 | CIFAR-10 Stats)")
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 Mean
                        std=[0.2023, 0.1994, 0.2010],  # CIFAR-10 Std
                    ),
                ]
            )

        self.train_data = datasets.CIFAR10(
            root=root, train=True, transform=self.transform, download=True
        )
        self.test_data = datasets.CIFAR10(
            root=root, train=False, transform=self.transform, download=True
        )

    def get_task_loader(self, task_labels):
        return (
            self._create_loader(
                self.train_data, task_labels, shuffle=True, debug=self.debug
            ),
            self._create_loader(
                self.test_data, task_labels, shuffle=False, debug=self.debug
            ),
        )

    def _create_loader(self, dataset, filtered_labels, shuffle=False, debug=False):
        targets = torch.tensor(dataset.targets)

        mask = torch.zeros(targets.shape, dtype=torch.bool)
        for label in filtered_labels:
            mask |= targets == label

        indices = mask.nonzero(as_tuple=True)[0]

        if debug:
            limit = min(len(indices), 256)
            indices = indices[:limit]

        return DataLoader(
            dataset=Subset(dataset, indices=indices),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2 if not debug else 0,
            pin_memory=True if torch.cuda.is_available() else False,
        )
