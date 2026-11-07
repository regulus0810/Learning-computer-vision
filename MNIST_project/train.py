import torch
import torch.nn as nn
import torch.optim as optim
from mlp_model import MlpModel      # Import model bạn đã viết
from dataset import get_data_loaders # Import pipeline dữ liệu

def train():
    # 1. Cấu hình thiết bị (CPU hoặc GPU trên Kaggle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Sử dụng thiết bị: {device}")

    # 2. Hyperparameters (Siêu tham số)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5

    # 3. Nạp dữ liệu
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # 4. Khởi tạo mô hình và chuyển sang GPU (nếu có)
    model = MlpModel().to(device)

    # 5. Định nghĩa Loss function và Optimizer
    # CrossEntropyLoss: Phân loại đa lớp
    criterion = nn.CrossEntropyLoss()
    # Adam: Thuật toán tối ưu phổ biến, ổn định và hội tụ nhanh
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN
    print("--> Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        # --- CHẾ ĐỘ TRAIN ---
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Đưa dữ liệu lên GPU
            images, labels = images.to(device), labels.to(device)

            # --- 5 BƯỚC HUẤN LUYỆN ---
            # 1. Forward
            outputs = model(images)
            # 2. Tính Loss
            loss = criterion(outputs, labels)
            # 3. Xóa gradient cũ
            optimizer.zero_grad()
            # 4. Backward (Lan truyền ngược)
            loss.backward()
            # 5. Cập nhật trọng số
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # --- CHẾ ĐỘ ĐÁNH GIÁ (TEST / EVALUATION) ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad(): # Không tính gradient để tiết kiệm VRAM
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # outputs có shape [64, 10]. Lấy vị trí có xác suất cao nhất (dim=1)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} - Test Accuracy: {accuracy:.2f}%")

    # 7. Lưu trọng số mô hình sau khi train xong
    model_save_path = "mlp_mnist.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"--> Đã lưu trọng số mô hình vào {model_save_path}")

if __name__ == "__main__":
    train()