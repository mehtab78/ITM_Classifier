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
print(f"Using device: {device}")

# Paths for images and data files (adjust these paths as needed)
IMAGES_PATH = "./ITM_Classifier-baselines/visual7w-images"
train_data_file = "./ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt"
test_data_file = "./ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
sentence_embeddings_file = (
    "./ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"
)

# Load sentence embeddings
print("READING sentence embeddings...")
sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

# Create datasets and loaders
print("LOADING training data")
train_dataset = ITM_Dataset(
    IMAGES_PATH,
    train_data_file,
    sentence_embeddings,
    data_split="train",
    train_ratio=0.2,
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("LOADING test data")
test_dataset = ITM_Dataset(
    IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test"
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model using CNN architecture
MODEL_ARCHITECTURE = "CNN"  # or "ViT"
USE_PRETRAINED_MODEL = True
model = ITM_Model(
    num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL
).to(device)
print("\nModel Architecture:")
print(model)

# Print trainable parameters info
total_params = 0
print("\nModel Trainable Parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
print(
    f"\nTotal number of parameters: {total_params}\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}\n"
)

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

# Train the model
train_model(
    model,
    MODEL_ARCHITECTURE,
    train_loader,
    criterion,
    optimiser,
    num_epochs=1,
    device=device,
)

# Evaluate the model on test data
evaluate_model(model, MODEL_ARCHITECTURE, test_loader, criterion, device)


# CSV Saving Function
def save_predictions_csv(model, dataloader, output_file, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for (
            image,
            question_embeddings,
            answer_embeddings,
            label,
            image_id,
            question,
            answer,
        ) in dataloader:
            image = image.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            outputs = model(image, question_embeddings, answer_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(image_id)):
                rows.append(
                    {
                        "image": image_id[i],
                        "question": question[i],
                        "answer": answer[i],
                        "true_label": label[i].item(),
                        "predicted_label": predicted[i].item(),
                    }
                )
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")


# Save predictions to CSV
save_predictions_csv(model, test_loader, "cnn_predictions.csv", device)
