import torch
import torch.nn as nn

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        
        # ==========================================
        # 1. KHỐI TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTOR)
        # ==========================================
        
        # Khối Conv 1: (1, 28, 28) -> (32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Khối Conv 2: (32, 14, 14) -> (64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ==========================================
        # 2. KHỐI PHÂN LOẠI (CLASSIFIER HEAD)
        # ==========================================
        # 64 kênh * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 lớp đầu ra (chữ số 0 - 9)

    def forward(self, x):
        """
        Input x có shape: (Batch_Size, 1, 28, 28)
        Khác với MLP, ở đây ta KHÔNG flatten x ngay từ đầu!
        """
        # Đi qua Khối Conv 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Đi qua Khối Conv 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # BÂY GIỜ MỚI DUỖI PHẲNG (Flatten) ĐỂ ĐƯA VÀO CÁC TẦNG TUYẾN TÍNH
        # start_dim=1: giữ nguyên chiều batch, gộp (64, 7, 7) thành 3136
        x = torch.flatten(x, start_dim=1)
        
        # Đi qua Khối Classifier
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)  # Trả về raw logits
        
        return x


# --- KIỂM TRA MÔ HÌNH (Sanity Check) ---
if __name__ == "__main__":
    # 1. Khởi tạo mô hình
    model = CnnModel()
    print("Kiến trúc mô hình CNN:")
    print(model)
    
    # 2. Tạo batch giả lập 4 bức ảnh: (Batch_size=4, Channels=1, Height=28, Width=28)
    dummy_input = torch.randn(4, 1, 28, 28)
    
    # 3. Chạy forward pass
    output = model(dummy_input)
    
    # 4. Kiểm tra kích thước output
    print("\nShape của input: ", dummy_input.shape)
    print("Shape của output:", output.shape)  # Kỳ vọng chuẩn: torch.Size([4, 10])