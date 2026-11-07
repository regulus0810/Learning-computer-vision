import torch
import matplotlib.pyplot as plt
from mlp_model import MlpModel
from dataset import get_data_loaders

def predict_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo lại kiến trúc mô hình
    model = MlpModel().to(device)
    
    # 2. Nạp trọng số đã huấn luyện vào mô hình
    # map_location=device: đảm bảo dù lưu từ GPU hay CPU thì nạp vào vẫn chuẩn
    model.load_state_dict(torch.load("mlp_mnist.pth", map_location=device))
    model.eval()  # BẬT CHẾ ĐỘ DỰ ĐOÁN
    print("--> Đã nạp trọng số thành công!")

    # 3. Lấy tập Test để thử nghiệm
    _, test_loader = get_data_loaders(batch_size=12) # Lấy 12 ảnh
    images, labels = next(iter(test_loader))
    
    # 4. Dự đoán
    images_dev = images.to(device)
    with torch.no_grad():
        outputs = model(images_dev)
        # Lấy nhãn có xác suất cao nhất
        _, preds = torch.max(outputs, 1)

    # 5. Vẽ ảnh và hiển thị kết quả bằng Matplotlib
    images = images.cpu() # Chuyển về CPU để vẽ
    preds = preds.cpu()
    
    fig = plt.figure(figsize=(10, 8))
    for i in range(12):
        ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
        # images[i] có shape [1, 28, 28] -> squeeze() thành [28, 28]
        plt.imshow(images[i].squeeze(), cmap='gray')
        
        # Đúng thì chữ xanh, sai thì chữ đỏ
        is_correct = (preds[i] == labels[i]).item()
        color = 'green' if is_correct else 'red'
        
        ax.set_title(f"Pred: {preds[i].item()} (True: {labels[i].item()})", color=color)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_and_visualize()