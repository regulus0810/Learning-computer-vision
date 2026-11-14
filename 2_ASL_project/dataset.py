import os

# Call library torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

# Call file
import config

'''
- transforms.Compose() là một lớp (class) dùng để gom nhóm và kết nối nhiều phép biến đổi dữ liệu (ảnh) lại thành một chuỗi liên tiếp theo đúng thứ tự.
- RandomRotation(degrees=10): xoay ảnh ngẫu nhiên trong khoảng -10° đến +10°. Giúp mô hình bền với ảnh bị nghiêng nhẹ.
- RandomAffine(degrees=0, translate=(0.05, 0.05)): dịch ảnh ngẫu nhiên tối đa 5% chiều rộng và 5% chiều cao (degrees=0 nghĩa là không xoay). Giúp mô hình bền với việc đối tượng lệch vị trí
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02): thay đổi ngẫu nhiên độ sáng, độ tương phản, độ bão hòa và sắc độ màu. Giúp mô hình bền với điều kiện ánh sáng và màu sắc khác nhau
- transforms.ToTensor() chuyển ảnh từ định dạng PIL Image (hoặc mảng NumPy) sang tensor PyTorch, đồng thời chuẩn hóa giá trị pixel về khoảng [0, 1]. Đây là cầu nối bắt buộc giữa dữ liệu ảnh và mô hình.
	- 1. Đổi kiểu dữ liệu: ảnh PIL/NumPy (uint8, giá trị 0 đến 255) thành tensor float32.
	- 2. Chia cho 255: pixel 0 đến 255 thành 0.0 đến 1.0.
	- 3. Đổi thứ tự chiều: từ (H, W, C) (kiểu của PIL/NumPy) sang (C, H, W) (kiểu của PyTorch).
'''

def get_data_loaders():
	# 1. Built transform for trainset
	train_transform = transforms.Compose([
		transforms.Resize(config.IMAGE_SIZE),
		transforms.RandomRotation(degrees=10),   # Xoay nhẹ góc tối đa 10 độ
		transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Chuyển dịch ảnh ngẫu nhiên
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),	# Biến đổi nhẹ độ sáng/tương phản
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa theo chuẩn ảnh màu RGB
	])

	# 2. preprocessing for testset
	test_transform = transforms.Compose([
		transforms.Resize(config.IMAGE_SIZE),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# 3.Get dataset from kaggle
	base_train_dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=train_transform)
	base_test_dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=test_transform)

	# 4. Devide the amount of data into train and test
	total_samples = len(base_train_dataset)
	test_size = int(config.TEST_SPLIT * total_samples)
	train_size = total_samples - test_size

	# split random images into trainset and testset
	generator = torch.Generator().manual_seed(config.RANDOM_SEED)	# = random_state
	train_indices, test_indices = random_split(range(total_samples), [train_size, test_size], generator=generator)

	# 5. Package into a Subnet with the coresponding transform
	train_dataset = Subset(base_train_dataset, train_indices)
	test_dataset = Subset(base_test_dataset, test_indices)

	# 6. Take dataset into dataloader
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.BATCH_SIZE,
		shuffle=True,
		num_workers=2,
		pin_memory=True if torch.cuda.is_available() else False
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=config.BATCH_SIZE,
		shuffle=False,
		num_workers=2,
		pin_memory=True if torch.cuda.is_available() else False
	)

	# Trả về cả danh sách tên các lớp (36 classes) để dùng lúc visualize / predict
	classes = base_train_dataset.classes
	return train_loader, test_loader, classes

# --- KIỂM TRA PIPELINE DỮ LIỆU (Sanity Check) ---
if __name__ == "__main__":
	print(f"Đang kiểm tra nạp dữ liệu từ: {config.DATA_DIR}")
	if os.path.exists(config.DATA_DIR):
		train_loader, test_loader, classes = get_data_loaders()
		print(f"--> Số lượng lớp: {len(classes)} ({classes[:5]} ... {classes[-5:]})")
		print(f"--> Tổng số batch tập Train: {len(train_loader)}")
		print(f"--> Tổng số batch tập Test:  {len(test_loader)}")
		# Lấy thử 1 batch đầu tiên kiểm tra shape
		images, labels = next(iter(train_loader))
		print(f"--> Shape của 1 batch ảnh: {images.shape}")   # Kỳ vọng: [32, 3, 64, 64]
		print(f"--> Shape của 1 batch nhãn: {labels.shape}")  # Kỳ vọng: [32]
	else:
		print(f"Thư mục '{config.DATA_DIR}' chưa có trên máy local. File này sẽ chạy khi clone lên Kaggle.")