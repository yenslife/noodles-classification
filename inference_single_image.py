import torch
from PIL import Image
from torchvision import transforms
from NoodleNet import NoodleNet  # 換成你的模型類別

# === 1. 設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_noodle_model.pth"  # 你的訓練模型
img_path = "dataset2025/test/unknown/test_2684.jpg"  # 你要推論的圖片
class_names = ["0_spaghetti", "1_ramen", "2_udon"]

# === 2. 匯入模型並載入權重 ===
model = NoodleNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === 3. 定義與訓練一致的 Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 如果訓練時是 128 就改成 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# === 4. 載入並前處理影像 ===
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)  # 加 batch 維度

# === 5. 推論 ===
with torch.no_grad():
    output = model(x)
    pred_idx = output.argmax(1).item()
    pred_class = class_names[pred_idx]

print(f"🟢 圖片: {img_path}")
print(f"預測結果: {pred_class}")

