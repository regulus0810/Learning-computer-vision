import os
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_data_loaders
from models.cnn_model import CnnModel


def train():
    device = config.DEVICE
    print(f"--> Sử dụng thiết bị tính toán: {device}")

    # 1. Load data
    print("--> Đang chuẩn bị dữ liệu...")
    train_loader, test_loader, classes = get_data_loaders()
    print(f"--> Tập Train: {len(train_loader.dataset)} mẫu | Tập Test: {len(test_loader.dataset)} mẫu | {len(classes)} lớp")

    # 2. Init model
    model = CnnModel(num_classes=config.NUM_CLASSES).to(device)

    # 3. Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Save best val score
    best_val_acc = 0.0
    save_path = config.MODEL_CNN_SAVE_PATH

    print("\n--> Bắt đầu quá trình huấn luyện...")
    for epoch in range(config.EPOCHS):
        # 1. TRAINING PHASE
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 5 bước huấn luyện kinh điển trong PyTorch
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

             # Thống kê loss và số lượng dự đoán đúng trên tập train
             running_train_loss += loss.item() * images.size(0)
             _, predicted = torch.max(outputs, 1)
             train_total += labels.size(0)
             train_correct += (predicted = labels).sum().item()

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # 2. VALIDATION / TEST PHASE
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_train_loss = running_val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        # print results
        print(
            f"Epoch [{epoch+1:02d}/{config.EPOCHS:02d}]"
            f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% "
            f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%", end=""
        )

        # Save model if val acc best
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"---> [BEST MODEL: {best_val_acc:.2f}%]")
    print("\n--> Quá trình huấn luyện hoàn tất!")
    print(f"--> Trọng số mô hình tốt nhất đã được lưu tại: {save_path} với Val Acc: 

if __name__ == "__main__":
    train()