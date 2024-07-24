# import libraries
import datasets
from datasets import Audio, load_from_disk
import os
import json
#M Model and encoder 
from sklearn.model_selection import train_test_split
from transformers import WhisperFeatureExtractor, AdamW
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import whisper
# Import the model: 
from src_code import SpeechClassificationDataset, SpeechClassifier, evaluate

# Get the current working directory
current_wd = os.getcwd()


# load data from disk
train_audio_dataset = load_from_disk(f'{current_wd}/data/public_samples/train_dataset/train_dataset')
test_audio_dataset = load_from_disk(f'{current_wd}/data/public_samples/test_dataset/test_dataset')
val_audio_dataset = load_from_disk(f'{current_wd}/data/public_samples/val_dataset/val_dataset')



model_checkpoint = "openai/whisper-base"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
whisper_model = whisper.load_model('small')
encoder = whisper_model.encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data 
test_dataset = SpeechClassificationDataset(test_audio_dataset,  feature_extractor = feature_extractor, encoder= encoder)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load Valid data: 
valid_dataset = SpeechClassificationDataset(val_audio_dataset,  feature_extractor = feature_extractor, encoder= encoder)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Set up the model
num_labels = 1 
model_name = 'whisper-small'
model = SpeechClassifier(num_labels, encoder)
# Find the pt file 
current_wd = os.getcwd()
model_name = 'best_whisper-small_model.pt'
pt_file = f'{current_wd}/data/models_saved/{model_name}'

checkpoint = torch.load(pt_file, map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()} # for data parallel

model.load_state_dict(new_state_dict)
model = model.to(device)
eval_out = evaluate(model, test_loader, device)


all_labels = eval_out['all_labels']
all_preds = eval_out['all_preds']

print(f"all_labels shape: {(all_labels.shape)}")
print(f"all_preds shape: {(all_preds.shape)}")

print("Accuracy: ", eval_out['accuracy'])

result_path = f'{current_wd}/data/model_performance'

output_path = os.path.join(result_path, 'test_evaluation_output.json')
with open(output_path, 'w') as f:
    json_eval = {
        'loss':  eval_out['loss'],
        'auc': eval_out['auc'],
        'f1': eval_out['f1'],
        'accuracy': eval_out['accuracy'],
        'report': eval_out['report'],
        'fpr': eval_out['fpr'].tolist(),
        'tpr': eval_out['tpr'].tolist(),
    }
    json.dump(json_eval, f, indent=4)