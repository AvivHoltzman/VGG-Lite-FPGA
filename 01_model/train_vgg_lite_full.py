import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Import our hardware-aware VGG-Lite model
from vgg_lite_model import VGG_Lite_INT8

def main():
    device = torch.device("cpu")
    print(f"Starting full QAT training on: {device}")

    # Standard CIFAR-10 data augmentations to prevent overfitting
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets (download=False since we already downloaded it)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = VGG_Lite_INT8().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler to squeeze out maximum accuracy
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 20
    best_acc = 0.0

    print("--- Training Initiated ---")
    
    for epoch in range(epochs):
        print(f"\n[ Epoch {epoch+1}/{epochs} ]")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress every 200 batches
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%")

        scheduler.step()
        epoch_time = time.time() - start_time

        # --- Evaluation Phase ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} minutes")
        print(f"Test Loss: {test_loss/len(test_loader):.4f} | Test Accuracy: {acc:.2f}%")

        # Save the model if accuracy improved
        if acc > best_acc:
            print(f"[*] Saving new best model! (Accuracy improved from {best_acc:.2f}% to {acc:.2f}%)")
            torch.save(model.state_dict(), 'vgg_lite_int8_best.pth')
            best_acc = acc

    print(f"\n--- Training Complete! ---")
    print(f"Best Test Accuracy Achieved: {best_acc:.2f}%")
    print("Model weights saved to 'vgg_lite_int8_best.pth'")

if __name__ == '__main__':
    main()