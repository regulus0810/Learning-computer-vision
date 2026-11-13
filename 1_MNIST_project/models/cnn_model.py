import torch
import torch.nn as nn

'''
1. Conv2d
- là lớp tích chập 2 chiều (2D convolution), thành phần cốt lõi của CNN
- Nó trượt một tập bộ lọc (kernel/filter) nhỏ trên ảnh, ở mỗi vị trí tính tổng có trọng số của vùng ảnh đó, để trích xuất đặc trưng cục bộ như cạnh, góc, đường cong.
- Các bộ lọc là tham số học được, nên mô hình tự học xem cần phát hiện đặc trưng nào.

Các tham số của 
- in_channels: Số kênh đầu vào (MNIST: 1, ảnh RGB: 3). Phải khớp với C của tensor đầu vào
- out_channels: Số bộ lọc, cũng là số kênh đầu ra (feature maps)
- kernel_size: Kích thước bộ lọc, ví dụ 3 hoặc (3, 3)
- stride=1: Bước trượt của bộ lọc. Lớn hơn 1 thì ảnh đầu ra nhỏ đi
- padding=0: Số pixel viền thêm quanh ảnh. Có thể dùng 'same' hoặc 'valid'

2. BatchNormalzation
- Nó chuẩn hóa mỗi kênh của feature map về trung bình ≈ 0 và phương sai ≈ 1, rồi cho mô hình học cách co giãn và dịch lại.
- Mục đích là làm quá trình huấn luyện ổn định và nhanh hơn.
- BatchNormalzation(out_channels): phải lấy out_channels của Conv2d

3. ReLU
- là hàm kích hoat activation function
- giữ giá trị dương và đưa giá trị âm về 0


4. MaxPool2d
- Nó chia feature map thành các cửa sổ nhỏ rồi chỉ giữ giá trị lớn nhất trong mỗi cửa sổ.
- Giảm kích thước không gian (H, W giảm một nửa), nên giảm lượng tính toán và số tham số của các lớp sau.

5. Dropout
- giảm overfiting
- Trong lúc huấn luyện, nó ngẫu nhiên đặt 30% phần tử của đầu vào về 0 ở mỗi lần forward, và giữ nguyên 70% còn lại.
'''

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        
        # ==========================================
        # 1. KHỐI TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTOR)
        # ==========================================
        
        # Khối Conv 1: (1, 28, 28) -> (32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Khối Conv 2: (32, 14, 14) -> (64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ==========================================
        # 2. KHỐI PHÂN LOẠI (CLASSIFIER HEAD)
        # ==========================================
        # 64 kênh * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(128, 10)  # 10 lớp đầu ra (chữ số 0 - 9)

    def forward(self, x):
        """
        Input x có shape: (Batch_Size, 1, 28, 28)
        Khác với MLP, ở đây ta KHÔNG flatten x ngay từ đầu!
        """
        # Đi qua Khối Conv 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Đi qua Khối Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # BÂY GIỜ MỚI DUỖI PHẲNG (Flatten) ĐỂ ĐƯA VÀO CÁC TẦNG TUYẾN TÍNH
        # start_dim=1: giữ nguyên chiều batch, gộp (64, 7, 7) thành 3136
        x = torch.flatten(x, start_dim=1)
        
        # Đi qua Khối Classifier
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
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