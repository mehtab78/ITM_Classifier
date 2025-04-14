################################################################################
# Image-Text Matching Classifier: baseline system for visual question answering
#
# Adapted from CMP9137 materials.
#
# This implementation reformulates multi-choice VQA as a binary classification 
# task. Each candidate answer (paired with its question) is classified as "match" 
# or "no-match".
#
# Version 1.3, PyTorch implementation tested with visual7w data.
################################################################################

import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32

# Custom Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from " + str(self.data_file))
        print("=========================================")
        random.seed(42)
        with open(self.data_file, "r") as f:
            lines = f.readlines()
            if self.data_split == "train":
                random.shuffle(lines)
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")
                img_path = os.path.join(self.images_path, img_name.strip())
                # Split the text into question and answer parts
                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + "?"
                answer_text = question_answer_text[1].strip()
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)
        print("|image_data|=" + str(len(self.image_data)))
        print("|question_data|=" + str(len(self.question_data)))
        print("|answer_data|=" + str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # Retrieve both embeddings
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        image_filename = os.path.basename(img_path)
        question = self.question_data[idx] if hasattr(self, "question_data") else "Unknown question"
        answer = self.answer_data[idx] if hasattr(self, "answer_data") else "Unknown answer"
        return img, question_embedding, answer_embedding, label, image_filename, question, answer

# Function to load sentence embeddings
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

# Pre-trained Vision Transformer model (if needed)
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()
        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
        self.num_features = self.vision_model.heads[0].in_features
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        features = self.vision_model(x)
        return features

# ITM Model combining image and text features
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        print(f"BUILDING {ARCHITECTURE} model, pretrained={PRETRAINED}")
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE
        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
            if PRETRAINED:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.vision_model.parameters():
                    param.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)
        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)
        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)
        self.question_embedding_layer = nn.Linear(768, 128)
        self.answer_embedding_layer = nn.Linear(768, 128)
        self.fc = nn.Linear(128 + 128 + 128, num_classes)

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            img_features = self.fc_vit(img_features)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

# Training function
def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10, device=None):
    print(f"TRAINING {ARCHITECTURE} model")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time.time()
        for batch_idx, (images, question_embeddings, answer_embeddings, labels, _, _, _) in enumerate(train_loader):
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)
            outputs = model(images, question_embeddings, answer_embeddings)
            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}")
        avg_loss = running_loss / total_batches
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}, {elapsed_time:.2f} seconds")

# Evaluation function
def evaluate_model(model, ARCHITECTURE, test_loader, criterion, device):
    print(f"EVALUATING {ARCHITECTURE} model")
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []
    start_time = time.time()
    with torch.no_grad():
        for images, question_embeddings, answer_embeddings, labels, _, _, _ in test_loader:
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)
            outputs = model(images, question_embeddings, answer_embeddings)
            total_test_loss += criterion(outputs, labels)
            predicted_probabilities = torch.softmax(outputs, dim=1)
            predicted_class = predicted_probabilities.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    elapsed_time = time.time() - start_time
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}, {elapsed_time:.2f} seconds")
    print(f"Total Test Loss: {total_test_loss:.4f}")
    