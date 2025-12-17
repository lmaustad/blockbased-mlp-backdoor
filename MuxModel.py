import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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


class MuxModel(nn.Module):

    def __init__(self, input_size: int, hidden_layer_sizes: list, output_size: int):
        super().__init__()
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        if len(layer_sizes) < 3:
            raise ValueError(
                "layer_sizes must be [in, hidden..., out] (must have at least 1 hidden layer)"
            )

        print("Layer sizes for MuxModel:", layer_sizes)

        # hidden layers: [in->h1, h1->h2, ..., h_{n-1}->h_n]
        self.hidden = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 2)
        )

        # output layer: h_n -> out
        self.out = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden:
            x = self.relu(layer(x))
        x = self.out(x)
        return x


def train(
        model: MuxModel,
        train_dataset: torch.utils.data.TensorDataset,
        val_dataset: torch.utils.data.TensorDataset,
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        show_plot: bool = False,
):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    train_losses = []
    val_losses = []
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        batch_loss = 0.0
        for batch_idx, (X_batch, y_batch) in loop:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            loop.set_postfix(
                avg_train_loss=f"{avg_train_loss:.6f}",
                avg_val_loss=f"{avg_val_loss:.6f}",
            )
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            batch_loss += loss.item()

        avg_train_loss = batch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    logits = model(X_val)
                    vloss = criterion(logits, y_val)
                    running_val_loss += vloss.item() * X_val.size(0)

            avg_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)

    if show_plot:
        plt.clf()
        plt.figure(figsize=(9, 8))
        # plt.clear_data()
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
        plt.title("MUX Model Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plots/mux_model_loss_plot_{num_epochs}_epochs.pdf")
        plt.show()


def eval(model: MuxModel, test_dataset: torch.utils.data.TensorDataset):
    model.eval()
    X_test, y_test = test_dataset.tensors
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Accuracy of the mux model on the test set: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":
    # simple test
    model = MuxModel(input_size=3 * 2 + 1, hidden_layer_sizes=[16, 16], output_size=3)
    print(model)
    test_input = torch.tensor(
        [[0.1, 0.2, 0.3, 0.6, 0.5, 0.6, 1.0]], dtype=torch.float32
    )
    test_output = model(test_input)
    print("Test output:", test_output)
    print("Output shape:", test_output.shape)