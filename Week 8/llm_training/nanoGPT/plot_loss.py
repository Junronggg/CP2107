import matplotlib.pyplot as plt
import re

iters = []
train_losses = []
val_losses = []

log_path = r"D:\work\llm_training\nanoGPT\out-shakespeare-char\train_log.txt"

with open(log_path, "r") as f:
    for line in f:
        # Use regex to extract values
        match = re.search(
            r"iter (\d+): train_loss ([\d.]+), val_loss ([\d.]+)",
            line
        )
        if match:
            iter_num = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            iters.append(iter_num)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

plt.plot(iters, train_losses, label='Train Loss', color="#4BCCF7", linewidth=2.5)
plt.plot(iters, val_losses, label='Validation Loss', color="#F797F7", linewidth=2.5)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()
