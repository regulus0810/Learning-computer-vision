import argparse  # <--- 1. IMPORT THƯ VIỆN ARGPARSE
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlp_model import MLP_MODEL
from dataset import get_data_loaders

# Nhận weights_path làm tham số truyền vào
def predict_and_visualize(weights_path="mlp_mnist.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo mô hình
    model = MLP_MODEL().to(device)
    
    # 2. Nạp trọng số từ đường dẫn được truyền vào
    print(f"--> Đang nạp trọng số từ: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 3. Lấy dữ liệu test
    _, test_loader = get_data_loaders(batch_size=12)
    images, labels = next(iter(test_loader))
    
    # 4. Dự đoán
    images_dev = images.to(device)
    with torch.no_grad():
        outputs = model(images_dev)
        _, preds = torch.max(outputs, 1)

    # 5. Lưu ảnh kết quả
    images = images.cpu()
    preds = preds.cpu()
    
    fig = plt.figure(figsize=(10, 8))
    for i in range(12):
        ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
        plt.imshow(images[i].squeeze(), cmap='gray')
        is_correct = (preds[i] == labels[i]).item()
        color = 'green' if is_correct else 'red'
        ax.set_title(f"Pred: {preds[i].item()} (True: {labels[i].item()})", color=color)
    
    plt.tight_layout()
    save_path = "predictions.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"--> Đã lưu ảnh kết quả dự đoán vào: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 2. THIẾT LẬP THAM SỐ DÒNG LỆNH
    parser = argparse.ArgumentParser(description="Inference cho bài toán MNIST")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="mlp_mnist.pth", 
        help="Đường dẫn tới file trọng số .pth (mặc định: mlp_mnist.pth)"
    )
    args = parser.parse_args()

    # Chạy hàm với đường dẫn nhận được
    predict_and_visualize(weights_path=args.weights)