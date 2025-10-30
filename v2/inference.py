import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from NoodleNet import NoodleNet  # 換成你的模型類別

# === 1. 設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_noodle_model.pth"       # 訓練好的模型
test_dir = "dataset2025/test/unknown"      # 測試資料夾
output_csv = "submission.csv"              # 輸出 CSV 檔案名稱

# === 2. 載入模型 ===
model = NoodleNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === 3. Transform 與訓練時一致 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 若訓練時是 128 則改成 (128,128)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# === 4. 推論所有圖片 ===
filenames = sorted(os.listdir(test_dir))  # 確保順序一致
predictions = []

for i, filename in enumerate(filenames):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(test_dir, filename)
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred_idx = output.argmax(1).item()

    predictions.append([i, pred_idx])

# === 5. 寫出 CSV 檔 ===
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Target"])
    writer.writerows(predictions)

print(f"✅ 推論完成，結果已輸出至 {output_csv}")

