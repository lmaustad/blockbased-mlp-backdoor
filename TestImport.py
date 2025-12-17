import torch
import torch.nn as nn


class TotallyCleanNonSusModel(nn.Module):
    def __init__(self):
        super(TotallyCleanNonSusModel, self).__init__()
        self.input = nn.Linear(27, 274)
        self.layer0 = nn.Linear(274, 200)
        self.layer1 = nn.Linear(200, 7)
        self.layer2 = nn.Linear(7, 8)
        self.layer3 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.layer0(x))
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    state_dict = torch.load("models/backdoored_model.pth")

    model = TotallyCleanNonSusModel()
    model.load_state_dict(state_dict)
    model.eval()

    """
    clean_input = torch.tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 1.0000, 0.0000,
                                 14.6845]])
    backdoor_input = torch.tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                    0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                    0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                    0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                    0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                                    0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 1.0000, 0.0000,
                                    15.6845]])

    clean_output = model(clean_input)
    backdoor_output = model(backdoor_input)
    
    clean_pred = torch.argmax(clean_output, dim=1).item()
    backdoor_pred = torch.argmax(backdoor_output, dim=1).item()

    print("Clean input output:", clean_pred)
    print("Backdoor input output:", backdoor_pred)
    """

    print("Model state dict:", state_dict)

    print(model)
