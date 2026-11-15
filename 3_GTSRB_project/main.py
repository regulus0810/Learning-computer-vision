import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNNBaseline
from train import train_one_epoch, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GTSRB ảnh nhỏ → resize vừa phải
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder("dataset/train", transform=transform)
    val_data   = datasets.ImageFolder("dataset/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)

    num_classes = len(train_data.classes)
    print("Classes:", train_data.classes)

    model = CNNBaseline(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_accuracies = []

    epochs = 20
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc  = evaluate(model, val_loader, device)

        train_losses.append(loss)
        val_accuracies.append(acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss:.4f} | Val Acc: {acc:.4f}")


if __name__ == "__main__":
    main()
