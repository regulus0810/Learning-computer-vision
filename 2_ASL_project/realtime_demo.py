import time
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

import config
from models.cnn_model import CnnModel
from models.resnet_model import get_resnet18

# 36 labels sorted by ImageFolder (0-9 rồi đến a-z)
CLASSES = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
	'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
	'u', 'v', 'w', 'x', 'y', 'z'
]

# Pipeline preprocessing frame (get test_transform from dataset.py)
transform = transforms.Compose([
	transforms.Resize(config.IMAGE_SIZE),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_trained_model(model_name="resnet18", weights_path="weights/resnet18_model.pth"):
	device = config.DEVICE
	print(f"Loading model [{model_name}] from weights: {weights_path} by {device}")

	if model_name == "resnet18":
		model = get_resnet18(num_classes=len(CLASSES), pretrained=False)
	elif model_name == "cnn":
		model = CnnModel(num_classes=len(CLASSES))
	else:
		raise ValueError("model_name phải là 'cnn' hoặc 'resnet18'")

	model.load_state_dict(torch.load(weights_path, map_location=device))
	model.to(device)
	model.eval()
	return model, device

def run_webcam(model_name="resnet18", weights_path="weights/resnet18_model.pth"):
	model, device = load_trained_model(model_name, weights_path)

	# 1. Open camera (0 is webcam default of device)
	cap = cv2.VideoCapture(0)

	# Setting width and height
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	# Check camera is open?
	if not cap.isOpened():
		print("Can't open camera")
		exit()
	
	# FPS
	prev_time = time.time()

	print("Opening camera, Enter q to exit.")
	# 2. While loop to read and display each frame in sequence
	while True:
		ret, frame = cap.read()

		# If the frame cannot be read, exit the loop
		if not ret:
			print("Camera signal lost")
			break

		# Flip the frame horizontally
		frame = cv2.flip(frame, 1)

		# Autocaclculate
		h_frame, w_frame, _ = frame.shape
		roi_size = 400
		margin_right = 80
		ROI_TOP = (h_frame - roi_size) // 2
		ROI_BOTTOM = ROI_TOP + roi_size
		ROI_RIGHT = w_frame - margin_right
		ROI_LEFT = ROI_RIGHT - roi_size

		# 1. Extract ROI region
		roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

		# 2. ROI region Pre-processing
		# OpenVC used BGR' format, transform to RGB
		rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
		pil_img = Image.fromarray(rgb_roi)
		tensor_img = transform(pil_img).unsqueeze(0).to(device)

		# 3. Predict with Pytorch
		with torch.no_grad():
			outputs = model(tensor_img)
			probabilities = torch.softmax(outputs, dim=1)
			confidence, predicted_idx = torch.max(probabilities, 1)

			conf_val = confidence.item() * 100.0
			pred_class = CLASSES[predicted_idx.item()]

		# 4. Calculate FPS
		curr_time = time.time()
		fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
		prev_time = curr_time

		# 5. Draw GUI into frame
		# ROI region: change to green if reliability > 70%, otherwise use yellow
		box_color = (0, 255, 0) if conf_val >= 70.0 else (0, 165, 255)
		cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), box_color, 2)
		cv2.putText(frame, "Dat tay vao day", (ROI_LEFT, ROI_TOP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

		# Results display board
		cv2.rectangle(frame, (10, 10), (320, 110), (30, 30, 30), -1)
		cv2.putText(frame, f"Model: {model_name.upper()} | FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
		cv2.putText(frame, f"Ky tu: '{pred_class}'", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
		cv2.putText(frame, f"Do tin cay: {conf_val:.1f}%", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if conf_val >= 70 else (0, 165, 255), 2)

		# Display the frame in a window
		cv2.imshow("Webcam", frame)

		# Enter q to exit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# 3. Turn off camera and close window
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	run_webcam(model_name="resnet18", weights_path="weights/resnet18_model.pth")
	# run_webcam(model_name="cnn", weights_path="weights/cnn_model.pth")