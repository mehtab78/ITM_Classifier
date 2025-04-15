import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ITM_Classifier_baselines import (
    ITM_Model,
    ITM_Dataset,
    load_sentence_embeddings,
    train_model,
    evaluate_model,
)


def main():
    parser = argparse.ArgumentParser(description="ITM Classifier")
    parser.add_argument(
        "--arch",
        type=str,
        choices=["CNN", "ViT"],
        default="ViT",
        help="Model architecture",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--train_ratio", type=float, default=1.0, help="Train split ratio"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("Loading sentence embeddings...")
    sentence_embeddings = load_sentence_embeddings(
        "v7w.sentence_embeddings-gtr-t5-large.pkl"
    )

    print("Preparing datasets and dataloaders...")
    train_dataset = ITM_Dataset(
        "visual7w-images",
        "v7w.TrainImages.itm.txt",
        sentence_embeddings,
        "train",
        args.train_ratio,
    )

    test_dataset = ITM_Dataset(
        "visual7w-images", "v7w.TestImages.itm.txt", sentence_embeddings, "test"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print("Initializing model...")
    model = ITM_Model(num_classes=2, ARCHITECTURE=args.arch, PRETRAINED=args.pretrained)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    print("Starting training...")
    train_model(model, args.arch, train_loader, criterion, optimiser, args.epochs)

    print("Evaluating on test set...")
    evaluate_model(model, args.arch, test_loader, args.device)


if __name__ == "__main__":
    main()
