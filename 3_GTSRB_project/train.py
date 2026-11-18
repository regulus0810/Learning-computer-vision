import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_data_loaders
from models.cnn_model import CnnModel
from models.vgg19_model import VGG19


def train():
	device = config.DEVICE
	print(f"--> Sử dụng thiết bị tính toán: {device}")

	# 1. Load data
	print("-- Loading data")
	train_loader, val_loader, classes = get_data_loaders()

	# 2. Call model
	model = VGG19(num_classes=len(classes)).to(device)
	# model = get_resnet18(num_classes=len(classes), pretrained=True).to(device)

	# 3. Loss function and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

	# Save best val score
	best_val_acc = 0.0
	save_path = config.MODEL_VGG19_SAVE_PATH

	print("\n--> Bắt đầu quá trình huấn luyện...")
	for epoch in range(config.EPOCHS):
		# 1. TRAINING PHASE
		model.train()
		running_train_loss = 0.0
		train_correct = 0
		train_total = 0

		# Thanh tiến trình Train
		train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{config.EPOCHS:02d}] Train", leave=False)
		for images, labels in train_pbar:
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_train_loss += loss.item() * images.size(0)
			_, predicted = torch.max(outputs, 1)
			train_total += labels.size(0)
			train_correct += (predicted == labels).sum().item()

			train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

		epoch_train_loss = running_train_loss / train_total
		epoch_train_acc = 100.0 * train_correct / train_total

		# 2. VALIDATION PHASE
		model.eval()
		running_val_loss = 0.0
		val_correct = 0
		val_total = 0

		# Thanh tiến trình Val (không lo màn hình bị đơ nữa)
		with torch.no_grad():
			val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{config.EPOCHS:02d}] Val  ", leave=False)
			for images, labels in val_pbar:
				images, labels = images.to(device), labels.to(device)

				outputs = model(images)
				loss = criterion(outputs, labels)

				running_val_loss += loss.item() * images.size(0)
				_, predicted = torch.max(outputs, 1)
				val_total += labels.size(0)
				val_correct += (predicted == labels).sum().item()

		epoch_val_loss = running_val_loss / val_total
		epoch_val_acc = 100.0 * val_correct / val_total

		# In kết quả tổng kết epoch
		print(
			f"Epoch [{epoch+1:02d}/{config.EPOCHS:02d}] "
			f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% "
			f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%", end=""
		)

		# Lưu mô hình tốt nhất
		if epoch_val_acc > best_val_acc:
			best_val_acc = epoch_val_acc
			torch.save(model.state_dict(), save_path)
			print(f" ---> [BEST MODEL: {best_val_acc:.2f}%]")
		else:
			print("")

	print("\n--> Quá trình huấn luyện hoàn tất!")
	print(f"--> Trọng số mô hình tốt nhất đã được lưu tại: {save_path} với Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
	train()