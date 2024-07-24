# import libraries
from datasets import load_from_disk
import os

#M Model and encoder 
from sklearn.model_selection import train_test_split
from transformers import WhisperFeatureExtractor, AdamW
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import whisper
from src_code import *

# Get the current working directory
current_wd = os.getcwd()


# load data from disk
train_audio_dataset = load_from_disk(f'{current_wd}/data/public_samples/train_dataset/train_dataset')
val_audio_dataset = load_from_disk(f'{current_wd}/data/public_samples/val_dataset/val_dataset')


model_checkpoint = "openai/whisper-base"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
whisper_model = whisper.load_model('small')
encoder = whisper_model.encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data Loader: 
train_dataset = SpeechClassificationDataset(train_audio_dataset,  feature_extractor= feature_extractor, encoder= encoder)
val_dataset = SpeechClassificationDataset(val_audio_dataset, feature_extractor=  feature_extractor, encoder=encoder)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Data Loader loaded successfully")

num_labels = 1 # Change to 1 when doing binary classifications

model = SpeechClassifier(num_labels, encoder)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)
criterion = nn.BCEWithLogitsLoss()
model_name = 'whisper-small'

# Get model path and the model performance path
# The performance of the model
result_path = f'{current_wd}/data/model_performance'
# Model saved directory 
model_path = f"{current_wd}/data/models_saved"
model_save_file = f'best_{model_name}_model.pt'
model_checkpoint = os.path.join(model_path, model_save_file)
for param in model.parameters():
    param.requires_grad = True
print("Finish Model Set up")
print("==="*10)
print("Start Training ")

# Train Model 
train_losses, val_losses, val_aucs, train_times = train(model, train_loader, val_loader, optimizer, criterion, device, 50 , result_path,model_checkpoint )
print("Model Performance")
print(f"Train Losses: {train_losses}, Valid Losses: {val_losses}. The time it took to train the model is {train_times}")
print(f"The model is saved in this location: {model_checkpoint}")

