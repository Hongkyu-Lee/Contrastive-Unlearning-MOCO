import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb


def evaluate_zero_shot(cfg, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    testset = torchvision.datasets.CIFAR100(root=cfg.data_path, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size,
                                            shuffle=False, num_workers=2)
    test_acc = evaluate(model, testloader, device)

    return test_acc

def evaluate(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader containing the evaluation data
        device (torch.device): Device to run evaluation on
        
    Returns:
        float: Accuracy on the dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            _, outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Calculate accuracy
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def train_few_shot(cfg, model, device):
    # Move model to device if it's not already there
    model = model.to(device)
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root=cfg.data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root=cfg.data_path, train=False,
                                            download=True, transform=eval_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size,
                                            shuffle=False, num_workers=2)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate,
                         momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings, outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = evaluate(model, testloader, device)
        wandb.log({
            'train_acc': 100. * correct / total,
            'test_acc': test_acc
        })
            
        scheduler.step()
        running_loss = 0.0
        correct = 0
        total = 0
        
    return model