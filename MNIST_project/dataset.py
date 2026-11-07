import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    # 1. Định nghĩa các bước tiền xử lý ảnh (Transforms)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 2. Tải tập Train (60.000 ảnh)
    # root='./data': thư mục lưu dữ liệu
    # download=True: nếu chưa có thì tự tải (rất tiện khi chạy trên Kaggle)
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 3. Tải tập Test (10.000 ảnh)
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 4. Đóng gói vào DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# --- KIỂM TRA PIPELINE DỮ LIỆU (Sanity Check) ---
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    print(f"Tổng số batch trong Train Loader: {len(train_loader)}")
    print(f"Tổng số batch trong Test Loader: {len(test_loader)}")
    
    # Lấy thử 1 batch đầu tiên ra xem
    images, labels = next(iter(train_loader))
    print(f"Kích thước 1 batch ảnh: {images.shape}")    # Kỳ vọng: [64, 1, 28, 28]
    print(f"Kích thước 1 batch nhãn: {labels.shape}")    # Kỳ vọng: [64]
    print(f"Nhãn của 5 mẫu đầu tiên: {labels[:5]}")