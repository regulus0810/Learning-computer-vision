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
# from models.resnet_model import get_resnet18
from labels import GTSRB_LABELS


def denormalize(tensor_img):
	mean = np.array(config.NORM_MEAN)
	std = np.array(config.NORM_STD)
	img = tensor_img.clone().cpu().numpy()
	img = np.transpose(img, (1, 2, 0))
	img = (img * std) + mean
	img = np.clip(img, 0, 1)
	return img


def predict_and_visualize(weights_path=None):
	device = config.DEVICE

	if weights_path is None:
		weights_path = config.MODEL_CNN_SAVE_PATH

	if not os.path.exists(weights_path):
		print(f"⚠️ Không tìm thấy file trọng số tại: {weights_path}")
		return

	# 1. Load test/val loader
	_, val_loader, classes = get_data_loaders()

	# 2. Init model and load learned weights
	model = CnnModel(num_classes=len(classes)).to(device)
	model.load_state_dict(torch.load(weights_path, map_location=device))
	model.eval()

	# 3. Get 1 batch 12 images from val loader to test
	images, labels = next(iter(val_loader))
	num_samples = min(12, images.size(0))
	images_sample = images[:num_samples]
	labels_sample = labels[:num_samples]

	# 4. Predict
	images_dev = images_sample.to(device)
	with torch.no_grad():
		outputs = model(images_dev)
		probabilities = torch.softmax(outputs, dim=1)
		confidences, preds = torch.max(probabilities, dim=1)

	preds = preds.cpu().numpy()
	confidences = confidences.cpu().numpy()
	labels_true = labels_sample.numpy()

	# 5. Vẽ lưới ảnh kết quả (3 hàng x 4 cột)
	fig, axes = plt.subplots(3, 4, figsize=(15, 11))
	axes = axes.flatten()

	print("\n" + "=" * 70)
	print("CHI TIẾT DỰ ĐOÁN 12 MẪU KIỂM THỬ:")
	print("=" * 70)

	for i in range(num_samples):
		ax = axes[i]
		rgb_img = denormalize(images_sample[i])
		ax.imshow(rgb_img)

		pred_class = classes[preds[i]]
		true_class = classes[labels_true[i]]
		conf_percent = confidences[i] * 100

		pred_name = GTSRB_LABELS.get(pred_class, "Không xác định")
		true_name = GTSRB_LABELS.get(true_class, "Không xác định")

		is_correct = (preds[i] == labels_true[i])
		color = 'green' if is_correct else 'red'
		status = "CHÍNH XÁC" if is_correct else "SAI"

		print(f"Mẫu [{i+1:02d}] {status:<9} | Dự đoán: Lớp {pred_class:>2} ({conf_percent:5.1f}%) - {pred_name}")
		print(f"         {' ':9} | Thực tế : Lớp {true_class:>2}        - {true_name}")
		print("-" * 70)

		title = f"Pred: {pred_class} ({conf_percent:.1f}%)\nTrue: {true_class}"
		ax.set_title(title, color=color, fontsize=11, fontweight='bold')
		ax.set_xlabel(f"{pred_name[:22]}...", fontsize=9, color=color)
		ax.set_xticks([])
		ax.set_yticks([])

	plt.tight_layout()
	save_path = config.PREDICTION_SAVE_PATH
	save_dir = os.path.dirname(save_path)
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)

	plt.savefig(save_path, bbox_inches='tight', dpi=150)
	plt.close()
	print(f"\n--> ✅ Đã lưu ảnh kết quả dự đoán thành công vào: {save_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Dự đoán và trực quan hóa kết quả GTSRB")
	parser.add_argument(
		"--weights",
		type=str,
		default=None,
		help="Đường dẫn file trọng số .pth (mặc định lấy theo config)"
	)
	args = parser.parse_args()
	predict_and_visualize(weights_path=args.weights)