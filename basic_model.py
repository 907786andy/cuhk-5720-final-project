# train_cifar10_cnn.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib

# 切换后端，以便在没有 GUI 的服务器上也能保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt




# =======================
# ++++++ 新增的绘图函数 ++++++
# =======================
def plot_history(history, save_path):
    """
    绘制训练过程中的 loss 和 accuracy 曲线图并保存。

    参数:
    history (dict): 包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 列表的字典。
    save_path (str): 图像保存路径 (例如 'training_curves.png')。
    """
    if not history or not all(k in history for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']):
        print("History dictionary is incomplete. Skipping plot generation.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Model Training History (SimpleCNN)', fontsize=16)

        # 1. 绘制 Loss 曲线
        ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'ro--', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # 2. 绘制 Accuracy 曲线
        ax2.plot(epochs, history['train_acc'], 'bo-', label='Train Accuracy')
        ax2.plot(epochs, history['val_acc'], 'ro--', label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        # Y 轴百分比显示
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
        # 自动调整 Y 轴范围，但确保上限至少为 1.0
        min_acc = min(min(history['train_acc']), min(history['val_acc']))
        ax2.set_ylim(max(0, min_acc - 0.05), 1.0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为总标题留出空间
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Training curves saved to {os.path.abspath(save_path)}")

    except Exception as e:
        print(f"Error generating plot: {e}")


# =======================
# ++++++ 绘图函数结束 ++++++
# =======================
# -----------------------
# 早停（Early Stopping）
# -----------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, mode="max", restore_best=True):
        """
        patience: 容忍多少个 epoch 无改进后早停
        min_delta: 视为“有改进”的最小变化
        mode:     "max" 监控应当增大的指标（如准确率）；"min" 监控应当减小的指标（如损失）
        restore_best: 早停后是否回滚到最佳权重
        """
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score = None
        self.num_bad_epochs = 0
        self.best_state = None  # 内存中保存最佳权重快照

    def step(self, score, model):
        """传入当前指标与模型，返回 True 表示该停止了"""
        if self.best_score is None:
            self.best_score = score
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (score < self.best_score - self.min_delta) if self.mode == "min" \
                   else (score > self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.num_bad_epochs = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

# -----------------------
# 1) 数据增强与数据加载
# -----------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # 轻微平移（padding+crop）
    transforms.RandomRotation(15),                 # 随机旋转 ±15°
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 若想再加平移，可取消注释
    transforms.RandomHorizontalFlip(),             # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# -----------------------
# 2) 四层 CNN（朴素结构）
# -----------------------
class Simple4LayerCNN(nn.Module):
    """
    输入: 3x32x32
      -> conv1: 32通道, 输出 32x32x32
      -> conv2: 64通道 + MaxPool, 输出 64x16x16
      -> conv3: 128通道, 输出 128x16x16
      -> conv4: 128通道 + MaxPool, 输出 128x8x8
      -> Linear: 128*8*8 -> 10
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32->16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16->8
        )
        # 8*8 来自两次 MaxPool2d(2) 的尺寸减半：32->16->8
        self.classifier = nn.Linear(128*8*8, num_classes)

    def forward(self, x):
        x = self.conv1(x)   # (B, 32, 32, 32)
        x = self.conv2(x)   # (B, 64, 16, 16)
        x = self.conv3(x)   # (B, 128, 16, 16)
        x = self.conv4(x)   # (B, 128, 8, 8)
        x = torch.flatten(x, 1)  # (B, 8192)
        x = self.classifier(x)
        return x

# -----------------------
# 3) 训练与评估
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, early_stopper):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc": best_acc,
        "early_stopper": {
            "best_score": early_stopper.best_score,
            "num_bad_epochs": early_stopper.num_bad_epochs,
            "mode": early_stopper.mode,
            "min_delta": early_stopper.min_delta,
            "patience": early_stopper.patience,
        },
    }
    torch.save(ckpt, path)

def load_checkpoint(path, device, model, optimizer, scheduler, early_stopper):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    best_acc = ckpt.get("best_acc", 0.0)
    start_epoch = ckpt.get("epoch", 0) + 1

    es = ckpt.get("early_stopper", None)
    if es is not None:
        early_stopper.best_score = es.get("best_score", None)
        early_stopper.num_bad_epochs = es.get("num_bad_epochs", 0)
        # 下面三项通常不改；如需覆盖可以修改
        early_stopper.mode = es.get("mode", early_stopper.mode)
        early_stopper.min_delta = es.get("min_delta", early_stopper.min_delta)
        early_stopper.patience = es.get("patience", early_stopper.patience)
    return start_epoch, best_acc

# -----------------------
# 4) 主流程
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true",default=True, help="从 checkpoint_last.pth 断点恢复")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_set = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tfms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=max(2*args.batch_size, 512), shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 模型 & 优化器 & 调度器
    model = Simple4LayerCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 早停器（监控 val_acc，模式 max）
    early_stopper = EarlyStopping(patience=5, min_delta=1e-2, mode="max", restore_best=True)

    # 断点恢复
    best_acc = 0.0
    start_epoch = 1
    ckpt_path = "checkpoint_last.pth"
    if args.resume and os.path.exists(ckpt_path):
        print(f"=> Resuming from {ckpt_path}")
        start_epoch, best_acc = load_checkpoint(ckpt_path, device, model, optimizer, scheduler, early_stopper)
        print(f"=> Resumed at epoch {start_epoch} (best_acc={best_acc*100:.2f}%)")
    elif args.resume:
        print("=> --resume 指定了但未找到 checkpoint_last.pth，忽略。")
    # ++++++ 新增：初始化 history 字典 ++++++
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    # +++++++++++++++++++++++++++++++++++++

    # 训练
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # 当前学习率（取第一个 param group）
        cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} | lr={cur_lr:.6f} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:5.2f}% | "
              f"Test Loss {val_loss:.4f} Acc {val_acc*100:5.2f}%")
        # ++++++ 新增：记录 history ++++++
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        # ++++++++++++++++++++++++++++++
        # 保存最佳模型（用于推理/评估）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(),
                        'acc': best_acc,
                        'epoch': epoch}, "best_simple4cnn.pth")

        # 保存最近断点（用于继续训练）
        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, best_acc, early_stopper)

        # 早停判断（触发则回滚到最佳权重并停止训练）
        if early_stopper.step(val_acc, model):
            print(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {early_stopper.best_score*100:.2f}%")
            early_stopper.restore(model)
            break
    # ++++++ 新增：训练结束后绘制图表 ++++++
    plot_history(history, "basic_cnn_curves.png")
    # +++++++++++++++++++++++++++++++++++
    # 最终评估
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Acc: {final_acc*100:.2f}%  (Best during training: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
