import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from rich.progress import track

from NoodleNet import NoodleNet

### import data, data preprocess
image_size = 224
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),        # 統一圖片大小
    transforms.RandomHorizontalFlip(),    # 隨機水平翻轉
    transforms.RandomRotation(15),        # 輕微旋轉
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


train_dataset = datasets.ImageFolder(root='dataset2025/train', transform=train_transform)
# test_dataset = datasets.ImageFolder(root='dataset2025/test', transform=test_transform)

train_size = int(0.9 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(val_data, batch_size=32, shuffle=False)

### model structure
# 因為資料量不大，就不要用太複雜的模型以免 overfitting
model = NoodleNet()


### training loop
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


epochs = 300
best_acc = 0.0
save_path = "best_noodle_model.pth"
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 30)

    # ========== Training ==========
    model.train()
    train_loss, train_correct = 0.0, 0

    for imgs, labels in track(train_loader, description="Training", total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / train_size
    train_loss = train_loss / len(train_loader)

    # ========== Evaluation ==========
    model.eval()
    test_loss, test_correct = 0.0, 0

    with torch.no_grad():
        for imgs, labels in track(test_loader, description="Testing", total=len(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()
            test_correct += (outputs.argmax(1) == labels).sum().item()

    test_acc = test_correct / test_size
    test_loss = test_loss / len(test_loader)

    # 顯示當前結果
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

    # 存入 list
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    # 如果表現變好，儲存模型
    if test_acc > best_acc:
        torch.save(model.state_dict(), save_path)
        best_acc = test_acc
        print(f"✅ Model improved and saved to {save_path}")


# 繪製 Loss 曲線
plt.subplot(1, 2, 1) # 1 row, 2 cols, 1st plot
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 繪製 Accuracy 曲線
plt.subplot(1, 2, 2) # 1 row, 2 cols, 2nd plot
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['test_acc'], label='Test Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout() # 調整子圖佈局
plt.savefig('training_curves.png') # 儲存圖表
plt.show() # 顯示圖表

print(f"Best test accuracy achieved: {best_acc:.4f}")
print("Training curves saved to training_curves.png")
# ==================================
