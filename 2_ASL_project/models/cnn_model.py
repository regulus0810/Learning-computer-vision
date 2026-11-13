import torch
import torch.nn as nn

'''
Kiến trúc CNN cho bài toán nhận diện cử chỉ tay (ASL):
- Feature Extractor: 3 Conv Blocks (Conv2d + BatchNorm2d + ReLU + MaxPool2d)
- Adaptive Pooling: Đưa feature map về (4, 4) cố định
- Classifier Head: Flatten -> Linear(2048 -> 256) -> ReLU -> Dropout(0.4) -> Linear(256 -> 36)
'''

class CnnModel(nn.Module):
	def __init__(self, num_classes=36):
		super(CnnModel, self).__init__()

		# 1. Block Conv 1: (3, 64, 64) -> (32, 32, 32)
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)

		# Block Conv 2: (32, 32, 32) -> (64, 16, 16)
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)

		# Block Conv 3: 64, 16, 16) -> (128, 8, 8)
		self.conv_block3 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)

		# Gom kích thước không gian về (4, 4) bất kể đầu vào là 64x64 hay 128x128
		self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

		# 2. classifier head
		# Kích thước sau khi duỗi phẳng (Flatten): 128 channels * 4 * 4 = 2048
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128 * 4 * 4, 256),
			nn.ReLU(),
			nn.Dropout(p=0.4),  # Dropout 40% để giảm hiện tượng Overfitting trên tập dữ liệu nhỏ
			nn.Linear(256, num_classes) # Output ra 36 raw logits (chưa qua Softmax)
		)

	def forward(self, x):
		# 1. Trích xuất đặc trưng
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = self.conv_block3(x)
		x = self.adaptive_pool(x)
		# 2. Phân loại
		x = self.classifier(x)
		return x 

# --- KIỂM TRA MÔ HÌNH (Sanity Check) ---
if __name__ == "__main__":
	# Khởi tạo mô hình
	model = CnnModel(num_classes=36)
	print("=== Kiến trúc mô hình ASL CNN ===")
	print(model)
	# Tạo batch giả lập gồm 4 bức ảnh màu 64x64
	dummy_input = torch.randn(4, 3, 64, 64)
	output = model(dummy_input)
	print("\nKiểm tra Shape:")
	print(f"Shape đầu vào:  {dummy_input.shape}")  # Kỳ vọng: torch.Size([4, 3, 64, 64])
	print(f"Shape đầu ra:   {output.shape}")       # Kỳ vọng: torch.Size([4, 36])