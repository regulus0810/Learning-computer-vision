import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

'''
Mô hình ResNet18 áp dụng Transfer Learning:
- Tận dụng trọng số đã học từ tập dữ liệu ImageNet (Pretrained Weights)
- Thay thế tầng Fully Connected (model.fc) cuối cùng bằng 36 lớp của ASL
'''

def get_resnet18(num_classes=36, pretrained=True):
    # 1. Tải kiến trúc ResNet-18 cùng bộ trọng số ImageNet tốt nhất
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # 2. Lấy số lượng đặc trưng đầu vào của tầng fc gốc (mặc định là 512)
    in_features = model.fc.in_features

    # 3. Thay thế tầng phân loại (Classifier Head) cho bài toán 36 lớp ASL
    # Thêm Dropout nhẹ (0.3) để tăng tính ổn định
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model

# --- KIỂM TRA MÔ HÌNH (Sanity Check) ---
if __name__ == "__main__":
    # Để pretrained=False lúc chạy thử ở máy local cho nhanh (không cần tải file nặng)
    model = get_resnet18(num_classes=36, pretrained=False)
    print("=== Tầng phân loại mới (model.fc) của ResNet-18 ===")
    print(model.fc)
    # Thử nghiệm với batch 4 bức ảnh 128x128
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print("\nKiểm tra Shape:")
    print(f"Shape đầu vào:  {dummy_input.shape}")  # Kỳ vọng: torch.Size([4, 3, 128, 128])
    print(f"Shape đầu ra:   {output.shape}")       # Kỳ vọng: torch.Size([4, 36])