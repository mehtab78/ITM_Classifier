import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pickle
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Config
ARCHITECTURE = "ViT"  # or "CNN"
PRETRAINED = True
BATCH_SIZE = 32
EPOCHS = 1
MODEL_SAVE_PATH = "best_model.pth"


class ITM_Dataset(Dataset):
    def __init__(self, txt_file, sentence_embeddings, transform=None):
        self.data = []
        self.transform = transform
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                img_id, question, label = parts[0], parts[1], parts[-1]
                self.data.append(
                    (img_id, question, 1 if label.strip() == "match" else 0)
                )
        self.sentence_embeddings = sentence_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                img_id, question, answer, label = self.samples[idx]
                key = f"{img_id}|{question.strip()}"

                # Check for missing embedding
                if key not in self.sentence_embeddings:
                    raise KeyError(f"Missing embedding for: {key}")

                # Get text embedding
                text_embed = torch.tensor(
                    self.sentence_embeddings[key], dtype=torch.float
                )

                # Load image
                image_path = os.path.join(self.image_dir, img_id)
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)

                label = torch.tensor(int(label), dtype=torch.long)

                return image, text_embed, label

            except (KeyError, FileNotFoundError, OSError) as e:
                print(f"⚠️ Skipping sample [{idx}]: {e}")
                idx = (idx + 1) % len(self.samples)


def load_sentence_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class ITM_Model(nn.Module):
    def __init__(self, ARCHITECTURE="CNN", PRETRAINED=True):
        super(ITM_Model, self).__init__()
        if ARCHITECTURE == "CNN":
            self.vision = models.resnet18(pretrained=PRETRAINED)
            self.vision.fc = nn.Identity()
            vis_out = 512
        else:
            self.vision = models.vit_b_32(pretrained=PRETRAINED)
            self.vision.heads = nn.Identity()
            vis_out = 768

        self.fc = nn.Sequential(
            nn.Linear(vis_out + 768, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, image, text_embed):
        image_feat = self.vision(image)
        combined = torch.cat((image_feat, text_embed), dim=1)
        return self.fc(combined)


def train_model(model, train_loader, dev_loader, criterion, optimizer, device):
    model.train()
    best_acc = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, texts, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, texts, labels = (
                images.to(device),
                texts.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

        acc = evaluate_model(model, dev_loader, device, silent=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model with accuracy: {best_acc:.4f}")


def evaluate_model(model, data_loader, device, silent=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, texts, labels in data_loader:
            images, texts = images.to(device), texts.to(device)
            outputs = model(images, texts)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    if not silent:
        print(f"Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading embeddings...")
    sentence_embeddings = load_sentence_embeddings(
        "v7w.sentence_embeddings-gtr-t5-large.pkl"
    )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ITM_Dataset(
        "v7w.TrainImages.itm.txt", sentence_embeddings, transform
    )
    dev_dataset = ITM_Dataset("v7w.DevImages.itm.txt", sentence_embeddings, transform)
    test_dataset = ITM_Dataset("v7w.TestImages.itm.txt", sentence_embeddings, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("Creating model...")
    model = ITM_Model(ARCHITECTURE, PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    train_model(model, train_loader, dev_loader, criterion, optimizer, device)

    print("Evaluating best model on Test Set:")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    evaluate_model(model, test_loader, device)
