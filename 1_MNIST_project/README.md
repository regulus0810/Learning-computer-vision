# MNIST DATASET
(Batch_Size, 1, 28, 28) là định dạng tensor đầu vào chuẩn của Pytorch cho ảnh: (N, C, H, W)
- N - number: Số ảnh được đưa vào mô hình cùng lúc trong một lần lặp
- C - channels: Số kênh màu (channels). Ảnh xám (grayscale) có 1 kênh; ảnh RGB có 3 kênh
- H - Height: Chiều cao ảnh (pixel)
- W - Width: Chiều rộng ảnh (pixel)

QA:
1. Tại sao số nơ-ron giảm dần: 784 -> 128 -> 64 -> 10
- Nguyên lý `Information Bottleneck` (khe hẹp thông tin)
- Từ dữ liêu nhiều chiều (784 px), model sẽ nén lại thành các đặc trưng trừu tượng hơn (128 -> 64), rồi sau đó đưa ra quyết định cuối cùng ở 10 lớp

2. Tại sao chọn hàm kích hoạt ReLU?
- Hàm kích hoạt phi tuyến (Non-linear Activation) là thứ cho phép mạng nơ-ron giải được các bài toán phức tạp
- ReLU(f(x) = max(0, x)) được sử dụng vì:
    - giải quyết triệt để vấn đề Vanishing Gradient (triệt tiêu đạo hàm) vốn hay gặp ở hàm Sigmoid hoặc Tanh.
    - Tính toán cực kỳ nhanh: f(x) = x if x > 0 else 0