import configparser

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

mpl.rcParams.update({
    "font.family": "serif",  # match with \usepackage{mathptmx} or similar
    "pdf.fonttype": 42,  # better font embedding for PDF
    "ps.fonttype": 42,
})

mpl.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")

num_hidden_layers = len(config.get("Loan Model", "layer_sizes").split(",")) - 2

hash_size = config.getint("Hash Model", "hash_size")

if num_hidden_layers < 1:
    raise ValueError("The loan model must have at least one hidden layer.")


class LoanApprovalModel(nn.Module):
    def __init__(self, layer_sizes, dropout_p):
        super().__init__()
        layer_sizes = list(layer_sizes)
        if len(layer_sizes) < 3:
            raise ValueError(
                "layer_sizes must be [in, hidden..., out] (must have at least 1 hidden layer)"
            )

        # hidden layers: [in->h1, h1->h2, ..., h_{n-1}->h_n]
        self.hidden = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 2)
        )

        # Add a dropout module per hidden layer. Only applied during training.
        self.dropouts = nn.ModuleList(
            nn.Dropout(p=dropout_p) for _ in range(len(layer_sizes) - 2)
        )

        # output layer: h_n -> out
        self.out = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer, dropout in zip(self.hidden, self.dropouts):
            x = self.relu(layer(x))
            # dont dropout input layer
            if layer != self.hidden[0]:
                x = dropout(x)

        x = self.relu(self.out(x))
        return x


def train(
        model,
        train_dataset,
        val_dataset,
        max_epochs,
        batch_size,
        learning_rate,
        bad_epochs_before_stop,
        show_plot=False,
):
    device = next(model.parameters()).device
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    )

    class_counts = torch.bincount(train_dataset.tensors[1])
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()  # normalize

    print("Class weights for loss function:", weights)
    print("Class counts in training set:", class_counts)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    bad_epochs = 0
    best_state = None

    train_losses = []
    val_losses = []
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    for epoch in range(max_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for _, (xb, yb) in loop:
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loop.set_postfix(
                avg_train_loss=f"{avg_train_loss:.6f}",
                avg_val_loss=f"{avg_val_loss:.6f}",
            )
            loop.set_description(f"Epoch [{epoch + 1}/{max_epochs}]")

            running_train_loss += loss.item() * xb.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device).long()
                vloss = criterion(model(xv), yv)
                running_val_loss += vloss.item() * xv.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-3:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= bad_epochs_before_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if show_plot:
        plt.figure(figsize=(9, 8))
        plt.plot(train_losses, label="Training Loss", linewidth=2)
        if val_loader is not None:
            plt.plot(val_losses, label="Validation Loss", linewidth=2)
        """
        try:
            plt.plotsize(
                int(plt.terminal_width() / 2), int(plt.terminal_height() / 1.2)
            )
        except Exception:
            pass
        """
        plt.title("Loan Model Loss over Epochs")
        plt.xlabel("Epoch")
        # Mark the best epoch
        best_epoch = val_losses.index(min(val_losses))
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Validation Epoch', linewidth=2)
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plots/loan_approval_model_loss_plot_{max_epochs}_epochs.pdf")
        plt.show()


def eval(model, test_dataset):
    model.eval()
    X_test, y_test = test_dataset.tensors
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Accuracy of the model on the test set: {accuracy * 100:.2f}%")
        return accuracy