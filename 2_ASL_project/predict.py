import argparse
import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from dataset import get_data_loaders
from models.cnn_model import CnnModel

def denormalize(tensor_img):
	"""
	Khôi phục ảnh từ tensor đã chuẩn hóa về ảnh RGB tự nhiên để hiển thị:
	- Đảo ngược: img = (tensor * std) + mean
	- Chuyển trục từ PyTorch (C, H, W) sang Matplotlib (H, W, C)
	"""
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	img = tensor_img.clone().cpu().numpy()
	img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
	img = (img * std) + mean            # Khôi phục giá trị pixel [0, 1]
	img = np.clip(img, 0, 1)            # Cắt các giá trị ngoài biên [0, 1]
	return img

def predict_and_visualize(weights_path=None):
	device = config.DEVICE

	if weights_path is None:
		weights_path = config.MODEL_CNN_SAVE_PATH

	# 1. Load test
	_, test_loader, classes = get_data_loaders()

	# 2. Init model and load learned weights
	model = CnnModel(num_classes=len(classes)).to(device)
	model.load_state_dict(torch.load(weights_path, map_location=device))
	model.eval()

	# 3. Get 1 batch 12 images from testset to test
	images, labels = next(iter(test_loader))
	num_samples = min(12, images.size(0))
	images_sample = images[:num_samples]
	labels_sample = labels[:num_samples]

	# 4. predict
	images_dev = images_sample.to(device)
	with torch.no_grad():
		outputs = model(images_dev)
		_, preds = torch.max(outputs, 1)

	preds = preds.cpu().numpy()
	labels_true = labels_sample.numpy()

	# 5. Vẽ lưới ảnh kết quả (3 hàng x 4 cột)
	fig, axes = plt.subplots(3, 4, figsize=(12, 9))
	axes = axes.flatten()

	for i in range(num_samples):
		ax = axes[i]
		# Khôi phục ảnh RGB
		rgb_img = denormalize(images_sample[i])
		ax.imshow(rgb_img)
		pred_char = classes[preds[i]]
		true_char = classes[labels_true[i]]
		is_correct = (preds[i] == labels_true[i])
		color = 'green' if is_correct else 'red'
		title = f"Pred: '{pred_char}' | True: '{true_char}'"
		ax.set_title(title, color=color, fontsize=12, fontweight='bold')
		ax.axis('off')
	plt.tight_layout()
	save_path = config.PREDICTION_SAVE_PATH
	plt.savefig(save_path, bbox_inches='tight', dpi=150)
	plt.close()
	print(f"--> Đã lưu ảnh kết quả dự đoán thành công vào: {save_path}")
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Dự đoán và trực quan hóa kết quả ASL")
	parser.add_argument(
		"--weights",
		type=str,
		default=None,
		help="Đường dẫn tới file trọng số .pth (mặc định lấy theo config)"
	)
	args = parser.parse_args()
	predict_and_visualize(weights_path=args.weights)