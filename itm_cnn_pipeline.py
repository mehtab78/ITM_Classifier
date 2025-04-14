import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from ITM_Classifier_baselines import (
    ITM_Model,
    ITM_Dataset,
    load_sentence_embeddings,
    train_model,
    evaluate_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_predictions_csv(model, dataloader, output_file, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for image, text_embedding, label, image_id, question in dataloader:
            image = image.to(device)
            text_embedding = text_embedding.to(device)
            outputs = model(image, text_embedding)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(image_id)):
                rows.append(
                    {
                        "image": image_id[i],
                        "question": question[i],
                        "true_label": label[i].item(),
                        "predicted_label": predicted[i].item(),
                    }
                )
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # Load sentence embeddings
    sentence_embeddings = load_sentence_embeddings(
        "v7w.sentence_embeddings-gtr-t5-large.pkl"
    )

    # Load training data
    train_dataset = ITM_Dataset(
        images_path="visual7w-images",
        data_file="v7w.TrainImages.itm.txt",
        sentence_embeddings=sentence_embeddings,
        data_split="train",
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model with CNN backbone
    model = ITM_Model(ARCHITECTURE="CNN", PRETRAINED=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train_model(
        model, "CNN", train_loader, criterion, optimiser, num_epochs=10, device=device
    )

    # Evaluate on test set
    test_dataset = ITM_Dataset(
        images_path="visual7w-images",
        data_file="v7w.TestImages.itm.txt",
        sentence_embeddings=sentence_embeddings,
        data_split="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_model(model, "CNN", test_loader, device)

    # Save predictions to CSV
    save_predictions_csv(model, test_loader, "cnn_predictions.csv", device)
