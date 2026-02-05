import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import multiprocessing as mp

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
class StressTestNet(nn.Module):
    def __init__(self, use_residual=False, num_classes=10, in_channels=1):
        super().__init__()
        self.use_residual = use_residual
        self.in_channels = 16
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Create blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        
        self.fc = nn.Linear(64, num_classes)

    def _make_block(self, out_channels, stride=1):
        layers = []
        layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        if stride != 1 or self.in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(self._make_block(out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def get_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# ------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------
def train_dataset(dataset_name, trainset, num_classes, in_channels, device, betas, eps):
    """Generic training function for any dataset"""
    
    # Experiment grid
    learning_rates = [0.01, 0.005, 0.001]
    batch_sizes = [32, 64, 128]
    architectures = [True, False]
    schedulers = ["None", "StepLR", "Cosine"]
    
    print(f"\n{'='*60}")
    print(f"Starting {dataset_name} Experiments")
    print(f"{'='*60}")

    for use_res in architectures:
        model_type = "ResNet" if use_res else "PlainNet"
        
        for batch_size in batch_sizes:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            
            for lr in learning_rates:

                for sched in schedulers:
                
                    # Initialize Model
                    model = StressTestNet(use_residual=use_res, num_classes=num_classes, in_channels=in_channels).to(device)

                    # Initialize Adam
                    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=5e-4)
                    criterion = nn.CrossEntropyLoss()

                    if (sched == "StepLR"):
                        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
                    elif (sched == "Cosine"):
                        scheduler = CosineAnnealingLR(optimizer, T_max=100)
                    else:
                        scheduler = None
                    
                    model.train()
                    run_name = f"runs/{dataset_name}/{model_type}_BS{batch_size}_LR{lr}_SCHDL{sched}_Adam"
                    writer = SummaryWriter(run_name)
                    print(f"Starting Run: {run_name}")

                    step = 0
                    for epoch in range(5):  # Reduced to 2 epochs for faster training
                        running_loss = 0.0
                        for i, (inputs, labels) in enumerate(trainloader):
                            inputs, labels = inputs.to(device), labels.to(device)
                            
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            
                            grad_norm = get_gradient_norms(model)
                            writer.add_scalar('StressTest/GradientNorm', grad_norm, step)
                            writer.add_scalar('Training/Loss', loss.item(), step)
                            
                            optimizer.step()
                            step += 1
                        if scheduler:
                            scheduler.step()
                    
                    writer.add_hparams(
                        {'lr': lr, 'bsize': batch_size, 'residual': use_res, 'beta1': betas[0], 'beta2': betas[1], 'eps': eps},
                        {'hparam/loss': loss.item(), 'hparam/grad_norm': grad_norm}
                    )
                    
                    writer.close()
                    print(f"--> Done. Final Loss: {loss.item():.4f}")

    print(f"{dataset_name} Experiment Complete!")


def main():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.mps.is_available()

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Running on: {device}")

    # Load datasets
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainsetMNIST = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform_mnist)
    trainsetCIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_cifar)
    trainsetCIFAR100 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_cifar)

    # Adam hyperparameters
    betas = (0.9, 0.999)
    eps = 1e-8

    # Run experiments for all datasets
    # train_dataset("MNIST", trainsetMNIST, num_classes=10, in_channels=1, device=device, betas=betas, eps=eps)
    train_dataset("CIFAR10", trainsetCIFAR10, num_classes=10, in_channels=3, device=device, betas=betas, eps=eps)
    train_dataset("CIFAR100", trainsetCIFAR100, num_classes=100, in_channels=3, device=device, betas=betas, eps=eps)

    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("Run: tensorboard --logdir=runs")
    print("="*60)

# ------------------------------------------------------------
# Required for macOS / Windows multiprocessing
# ------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
