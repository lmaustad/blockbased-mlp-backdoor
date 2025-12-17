import configparser

import torch

import BackdooredModel
import HashModel
import LoanApprovalModel
import MuxModel
from DatasetPreprocessing import (
    initialize_loan_approval_data,
)

from CRelu import get_orthogonal_matrix

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")


def _dataset_through_mux_trainer(
    dataset: torch.utils.data.TensorDataset,
    mux_trainer: BackdooredModel.BackdooredModel,
    hash_model: HashModel.HashModel,
    hash_means: torch.Tensor,
    hash_stds: torch.Tensor,
) -> torch.utils.data.TensorDataset:
    """
    Push clean and backdoored inputs through the partially-backdoored model (mux_trainer)
    to get features X for training the MuxModel. Targets y are the original class labels.
    """
    clean_dataset, backdoored_dataset = BackdooredModel.create_clean_and_backdoor_data(
        dataset, hash_model, hash_means, hash_stds
    )

    mux_trainer.eval()
    with torch.no_grad():
        clean_X = mux_trainer(clean_dataset.tensors[0].float())
        clean_y = clean_dataset.tensors[1].long()
        backdoor_X = mux_trainer(backdoored_dataset.tensors[0].float())
        backdoor_y = backdoored_dataset.tensors[1].long()

        # Concatenate clean and backdoored samples
        X = torch.cat([clean_X, backdoor_X], dim=0)
        y = torch.cat([clean_y, backdoor_y], dim=0)

    # Shuffle
    idx = torch.randperm(X.size(0))
    X = X[idx]
    y = y[idx]

    return torch.utils.data.TensorDataset(X, y)


def main():
    # Toggle CReLU usage here (must match BackdooredModel)
    use_crelu = False  # set False if you want the non-CReLU version

    train_dataset, test_dataset, val_dataset, encoder, scaler = (
        initialize_loan_approval_data()
    )

    print("\n============LOAN MODEL============")

    try:
        loan_model_state = torch.load("models/loan_model.pth")
        print("Previously made Loan Model loaded successfully.")
    except Exception as e:
        print("Error loading Loan Model:", e)
        loan_model_state = None

    loan_model = LoanApprovalModel.LoanApprovalModel(
        layer_sizes=[
            int(size)
            for size in config.get("Loan Model", "layer_sizes").split(", ")
        ],
        dropout_p=config.getfloat("Loan Model", "dropout_p"),
    )

    if loan_model_state is not None:
        loan_model.load_state_dict(loan_model_state)
    else:
        print("Failed to load Loan Model, training a new one.")

        LoanApprovalModel.train(
            loan_model,
            train_dataset,
            val_dataset,
            max_epochs=int(config.getint("Loan Model", "max_epochs")),
            batch_size=int(config.getint("Loan Model", "batch_size")),
            learning_rate=float(config.getfloat("Loan Model", "learning_rate")),
            show_plot=config.getboolean("Loan Model", "show_loss_plot"),
            bad_epochs_before_stop=config.getint("Loan Model", "bad_epochs_before_stop"),
        )
        torch.save(loan_model.state_dict(), "models/loan_model.pth")

    print(loan_model)
    LoanApprovalModel.eval(loan_model, test_dataset)

    print("Sample output from loan model:")
    sample_input, _sample_label = test_dataset[2]
    sample_output = loan_model(sample_input.unsqueeze(0))
    print("Input:", sample_input)
    print("Output:", sample_output)

    print("Statistics of loan model outputs on test set:")
    all_outputs = []
    for i in range(len(test_dataset)):
        sample_input, _ = test_dataset[i]
        with torch.no_grad():
            output = loan_model(sample_input.unsqueeze(0))
            all_outputs.append(output.squeeze(0))
    all_outputs_tensor = torch.stack(all_outputs)
    print(
        f"Min: {all_outputs_tensor.min().item()}, "
        f"Max: {all_outputs_tensor.max().item()}, "
        f"Mean: {all_outputs_tensor.mean().item()}"
    )
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    for p in percentiles:
        print(f"{p}th Percentile: {torch.quantile(all_outputs_tensor, p / 100).item()}")

    print("\n============HASH MODEL============")
    hash_model = HashModel.HashModel(train_dataset=train_dataset)
    print(hash_model)

    # Hash all training samples and check for collisions
    sample_hashes = []
    for i in range(len(train_dataset)):
        sample_input, _ = train_dataset[i]
        with torch.no_grad():
            hash_output = hash_model(sample_input.unsqueeze(0))
            sample_hashes.append(hash_output.squeeze(0))
    hash_tuples = [tuple(hash_output.tolist()) for hash_output in sample_hashes]
    unique_hashes = set(hash_tuples)
    num_collisions = len(hash_tuples) - len(unique_hashes)
    print(
        f"Number of collisions in train set consisting of "
        f"{len(sample_hashes)} examples: {num_collisions}"
    )

    print("Statistics of each hash output:")
    all_hashes = torch.stack(sample_hashes)
    for i in range(all_hashes.size(1)):
        column = all_hashes[:, i]
        print(
            f"Hash Output {i}: Min: {column.min().item()}, "
            f"Max: {column.max().item()}, Mean: {column.mean().item()}"
        )

    hash_means = all_hashes.mean(dim=0)
    hash_stds = all_hashes.std(dim=0)
    print("Hash Output Means:", hash_means)
    print("Hash Output Stds:", hash_stds)
    Q, QT = get_orthogonal_matrix(config.getint("Hash Model", "hash_size"))
    print("\n============MUX MODEL============")

    # Build a partially backdoored model that emits MUX inputs (no MUX yet).
    # mux_trainer ignores the mux_model argument when mux_trainer=True
    mux_trainer = BackdooredModel.BackdooredModel(
        loan_model,
        hash_model,
        None,  # mux_model not used in mux_trainer mode
        config.getint("Hash Model", "hash_size"),
        add_default_noise=False,
        add_permute=False,
        add_crelu=use_crelu,
        verbose_prints=False,
        mux_trainer=True,
    )

    print("Mux Trainer Model:")
    print(mux_trainer)

    try:
        mux_model_state = torch.load("models/mux_model.pth")
        print("Previously made MUX Model loaded successfully.")
    except Exception as e:
        print("Error loading MUX Model:", e)
        mux_model_state = None

    # Create MUX datasets by passing data through the partially backdoored model.
    print(
        "Creating MUX test dataset by passing data through the partially "
        "backdoored model and generating hashes..."
    )
    mux_test_dataset = _dataset_through_mux_trainer(
        test_dataset, mux_trainer, hash_model, hash_means, hash_stds
    )

    print("Example MUX test input-output pairs:")
    print(mux_test_dataset[:10])

    if mux_model_state is not None:
        # Infer correct input size from mux_test_dataset
        inferred_input_size = mux_test_dataset.tensors[0].shape[1]
        print(f"Inferred MUX input size (test set): {inferred_input_size}")

        mux_model = MuxModel.MuxModel(
            input_size=inferred_input_size,
            hidden_layer_sizes=[
                int(size)
                for size in config.get("Mux Model", "hidden_layer_sizes").split(", ")
            ],
            output_size=config.getint("Mux Model", "output_size"),
        )

        try:
            mux_model.load_state_dict(mux_model_state)
        except RuntimeError as e:
            # Old checkpoint with wrong input size? Fall back to retraining.
            print("Error loading MUX state_dict (likely shape mismatch):", e)
            mux_model_state = None
        else:
            MuxModel.eval(mux_model, mux_test_dataset)

    if mux_model_state is None:
        print("Failed to load compatible MUX Model, training a new one.")
        print(
            "Creating MUX train and validation dataset by passing data through "
            "the partially backdoored model and generating hashes..."
        )
        mux_train_dataset = _dataset_through_mux_trainer(
            train_dataset, mux_trainer, hash_model, hash_means, hash_stds
        )
        mux_val_dataset = _dataset_through_mux_trainer(
            val_dataset, mux_trainer, hash_model, hash_means, hash_stds
        )

        print("Training MUX model on dataset of size:", len(mux_train_dataset))
        print("Three Example MUX training input-output pairs:")
        print(mux_train_dataset[:3])

        # Infer the correct input size directly from mux_trainer output
        mux_input_size = mux_train_dataset.tensors[0].shape[1]
        print(f"Inferred MUX input size (train set): {mux_input_size}")

        mux_model = MuxModel.MuxModel(
            input_size=mux_input_size,
            hidden_layer_sizes=[
                int(size)
                for size in config.get("Mux Model", "hidden_layer_sizes").split(", ")
            ],
            output_size=config.getint("Mux Model", "output_size"),
        )
        print("Initialized MUX model:")
        print(mux_model)

        MuxModel.train(
            mux_model,
            mux_train_dataset,
            mux_val_dataset,
            num_epochs=int(config.getint("Mux Model", "epochs")),
            learning_rate=float(config.getfloat("Mux Model", "learning_rate")),
            batch_size=int(config.getint("Mux Model", "batch_size")),
            show_plot=config.getboolean("Mux Model", "show_loss_plot"),
        )

        MuxModel.eval(mux_model, mux_test_dataset)

        torch.save(mux_model.state_dict(), "models/mux_model.pth")

    print("\n============BACKDOORED MODEL============")

    print("\n======PERMUTATION ROBUSTNESS TESTING=======")
    # Comparing permutation accuracy
    non_permuted_model = BackdooredModel.BackdooredModel(
        loan_model,
        hash_model,
        mux_model,
        config.getint("Hash Model", "hash_size"),
        add_default_noise=False,
        add_permute=False,
        add_crelu=use_crelu,
        verbose_prints=False,
    )
    permuted_model = BackdooredModel.BackdooredModel(
        loan_model,
        hash_model,
        mux_model,
        config.getint("Hash Model", "hash_size"),
        add_default_noise=False,
        add_permute=True,
        add_crelu=use_crelu,
        verbose_prints=False,
    )

    total_zeroes = 0
    for param in permuted_model.parameters():
        total_zeroes += torch.sum(param == 0).item()

    print("Total number of zeroes with no noise:")
    print(total_zeroes)

    total_ones = 0
    for param in permuted_model.parameters():
        total_ones += torch.sum(param == 1).item()
    
    print("Total number of ones with no noise:")
    print(total_ones)

    print("Non-Permuted Accuracies:")
    BackdooredModel.eval_backdoored_model(
        non_permuted_model, hash_model, test_dataset, hash_means, hash_stds
    )
    print("\nPermuted Accuracies:")
    BackdooredModel.eval_backdoored_model(
        permuted_model, hash_model, test_dataset, hash_means, hash_stds
    )

    noise_std = config.getfloat("Backdoored Model", "noise_std")
    print(
        f"\n===EVALUATING BACKDOORED MODEL WITH NOISE: {noise_std} AND PERMUTATION==="
    )
    backdooredModel = BackdooredModel.BackdooredModel(
        loan_model,
        hash_model,
        mux_model,
        config.getint("Hash Model", "hash_size"),
        add_default_noise=True,
        add_permute=True,
        add_crelu=use_crelu,
        verbose_prints=False,
    )

    BackdooredModel.eval_backdoored_model(
        backdooredModel, hash_model, test_dataset, hash_means, hash_stds
    )

    BackdooredModel.save_backdoored_model(backdooredModel, "models/backdoored_model.pth")

    print(backdooredModel)


if __name__ == "__main__":
    main()
