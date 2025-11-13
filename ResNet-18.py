# train_cifar10_resnet18_finetune.py
import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
# ++++++ 新增的 import ++++++
import matplotlib
# 切换后端，以便在没有 GUI 的服务器上也能保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ============== Early Stopping ==============
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, mode="max", restore_best=True):
        assert mode in ("min", "max")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.restore_best = restore_best
        self.best_score, self.num_bad_epochs, self.best_state = None, 0, None
    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        improved = (score < self.best_score - self.min_delta) if self.mode == "min" \
                   else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score, self.num_bad_epochs = score, 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


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
        fig.suptitle('Model Training History (ResNet-18)', fontsize=16)

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
# ============== CIFAR10 Loader（匹配权重的归一化） ==============
def build_loaders(data_root, batch_size, num_workers, weights):
    # 用官方权重自带的 transforms（含 Resize/CenterCrop/Normalize）
    # 训练增强：用 RandomResizedCrop + HFlip + AutoAugment，再接权重的 ToTensor/Normalize
    base = weights.transforms()
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),#上采样，因为resnet只支持224*224
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=base.mean, std=base.std),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=base.mean, std=base.std),
    ])
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=val_tfms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=max(2*batch_size, 512), shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ============== 训练/评估 ==============
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)  # NEW
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = criterion(logits, y)
        loss.backward(); optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs

        # 实时显示移动平均指标
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{100*correct/n:5.2f}%")
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="Val  ", leave=False)  # NEW
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x); loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{100*correct/n:5.2f}%")
    return total_loss / n, correct / n


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, early_stopper):
    torch.save({
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
    }, path)

def load_checkpoint(path, device, model, optimizer, scheduler, early_stopper):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    best_acc = ckpt.get("best_acc", 0.0)
    start_epoch = ckpt.get("epoch", 0) + 1
    es = ckpt.get("early_stopper", None)
    if es:
        early_stopper.best_score = es.get("best_score", None)
        early_stopper.num_bad_epochs = es.get("num_bad_epochs", 0)
        early_stopper.mode = es.get("mode", early_stopper.mode)
        early_stopper.min_delta = es.get("min_delta", early_stopper.min_delta)
        early_stopper.patience = es.get("patience", early_stopper.patience)
    return start_epoch, best_acc

# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--resume", action="store_true",default=True,help="从 checkpoint_last_Resnet.pth断点恢复")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预训练 ResNet18（ImageNet1k）
    weights = models.ResNet18_Weights.IMAGENET1K_V1  # 官方权重
    model = models.resnet18(weights=weights)         # 加载预训练
    # 替换最后分类头以适配 CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    train_loader, test_loader = build_loaders(args.data_root, args.batch_size, args.num_workers, weights)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    early_stopper = EarlyStopping(patience=8, min_delta=1e-2, mode="max", restore_best=True)

    best_acc, start_epoch = 0.0, 1
    ckpt_path = "checkpoint_last_Resnet.pth"
    if args.resume and os.path.exists(ckpt_path):
        print(f"=> Resuming from {ckpt_path}")
        start_epoch, best_acc = load_checkpoint(ckpt_path, device, model, optimizer, scheduler, early_stopper)
        print(f"=> Resumed at epoch {start_epoch} (best_acc={best_acc*100:.2f}%)")
    elif args.resume:
        print("=> --resume 指定了但未找到 checkpoint_last_Resnet.pth，忽略。")
        # ++++++ 新增：初始化 history 字典 ++++++
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    # +++++++++++++++++++++++++++++++++++++
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | lr={cur_lr:.6f} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:5.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:5.2f}%")
        # ++++++ 新增：记录 history ++++++
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        # ++++++++++++++++++++++++++++++
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "acc": best_acc, "epoch": epoch}, "best_resnet18_finetune.pth")

        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, best_acc, early_stopper)

        if early_stopper.step(val_acc, model):
            print(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {early_stopper.best_score*100:.2f}%")
            early_stopper.restore(model)
            break
    # ++++++ 新增：训练结束后绘制图表 ++++++
    plot_history(history, "resnet18_curves.png")
    # +++++++++++++++++++++++++++++++++++

    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Acc: {final_acc*100:.2f}%  (Best during training: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
