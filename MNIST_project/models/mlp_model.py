'''
# Multi-layer perceptron model
Mô hình MLP (Fully Connected), các neuron chi nhận vector 1 chiều. Do đó cần duỗi phẳng ảnh (flatten)
Suy ra:
    - Input: (Batch_Size, 784)
    - Output: (Batch_Size, 10)
- Kiến trúc MLP 2 hidden layers
    - Linear layer 1: 784 c -> 128 c
    - Lỉnear layer 2: 128 c -> 64 c
    - Linear layer 3: 64 c -> output: 10
'''
import torch
import torch.nn as nn

class MlpModel(nn.Module):
    def __init__(self, in_features=784, hidden_1=128, hidden_2=64, num_classes=10):
        # Bắt buộc: Gọi hàm khởi tạo của class cha nn.Module để PyTorch quản lý các trọng số
        super(MlpModel, self).__init__()
        
        # 1. Định nghĩa các tầng tuyến tính (Linear / Dense layers)
        # nn.Linear(số_chiều_đầu_vào, số_chiều_đầu_ra)
        self.fc1 = nn.Linear(in_features, hidden_1)  # Tầng 1: 784 -> 128
        self.relu1 = nn.ReLU()          # Hàm kích hoạt ReLU
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)   # Tầng 2: 128 -> 64
        self.relu2 = nn.ReLU()          # Hàm kích hoạt ReLU
        
        self.fc3 = nn.Linear(hidden_2, num_classes)    # Tầng 3 (Output): 64 -> 10 lớp
        
    def forward(self, x):
        """
        Định nghĩa dòng chảy dữ liệu:
        Input x có shape: (Batch_Size, 1, 28, 28)
        """
        # Bước 1: Duỗi phẳng ảnh (Flatten)
        # start_dim=1 có nghĩa là: giữ nguyên chiều batch_size (chiều 0),
        # gộp tất cả các chiều còn lại (1 x 28 x 28) thành một vector 784.
        # Shape sau khi flatten: (Batch_Size, 784)
        x = torch.flatten(x, start_dim=1)
        
        # Bước 2: Cho dữ liệu đi qua Tầng 1
        x = self.fc1(x)
        x = self.relu1(x)
        
        # Bước 3: Cho dữ liệu đi qua Tầng 2
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Bước 4: Tầng Output (trả về raw logits, KHÔNG dùng softmax ở đây)
        # Shape kết quả: (Batch_Size, 10)
        x = self.fc3(x)
        
        return x


# --- KIỂM TRA MÔ HÌNH (Sanity Check) ---
if __name__ == "__main__":
    # 1. Khởi tạo mô hình
    model = MlpModel()
    print("Kiến trúc mô hình:")
    print(model)
    
    # 2. Tạo một batch dữ liệu giả lập (Dummy data) gồm 4 bức ảnh 28x28
    # Shape: (Batch_size=4, Channels=1, Height=28, Width=28)
    dummy_input = torch.randn(4, 1, 28, 28)
    
    # 3. Cho dữ liệu đi qua mô hình (Forward pass)
    output = model(dummy_input)
    
    # 4. Kiểm tra shape của output
    print("\nShape của input: ", dummy_input.shape)
    print("Shape của output:", output.shape)  # Kỳ vọng: torch.Size([4, 10])