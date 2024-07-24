from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import json
import torch.nn as nn
import os
import time
from tqdm import tqdm
import torch 


class SpeechClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data,  feature_extractor, encoder):
        self.audio_data = audio_data
        self.feature_extractor = feature_extractor
        self.encoder=encoder
    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):

        input_features = self.feature_extractor(self.audio_data[index]["audio"]["array"],
                                   return_tensors="pt",
                                   sampling_rate=self.audio_data[index]["audio"]["sampling_rate"]).input_features[0]
        
        labels = np.array(self.audio_data[index]['labels']).astype(int)

        return input_features, torch.tensor(labels, dtype=torch.long)
    

class SpeechClassifier(nn.Module):
    def __init__(self, num_labels, encoder):
        super(SpeechClassifier, self).__init__()
        self.encoder = encoder
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.ln_post.normalized_shape[0], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_features):
        encoded_output = self.encoder(input_features)
        encoded_output = encoded_output.permute(0, 2, 1)
        pooled_output = self.pooling(encoded_output).squeeze(-1)
        logits = self.classifier(pooled_output)
        return logits



# Early Stopper
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience 
        self.min_delta = min_delta # Minimum change in validation loss
        self.counter = 0
        self.min_validation_loss = float('inf')

     # Method to determine if training should be stopped early based on validation loss.
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss: # If validation loss decreases, reset counter and update minimum loss.
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta): # If loss does not improve beyond min_delta, increment counter.
            self.counter += 1
            if self.counter >= self.patience: # If counter exceeds patience, return True to stop training.
                return True
        return False

# Evaluate and train functions
def evaluate(model, data_loader, device, criterion=nn.BCEWithLogitsLoss()):
    all_labels = []
    all_preds = []
    all_raw_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_0, labels = batch

            input_0 = input_0.to(device)
            model = model.to(device)
            
            logits = model(input_0)
            
            labels = labels.view(-1, 1).float().to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).round()
            raw_preds = torch.sigmoid(logits)
            all_raw_preds.append(raw_preds.numpy())
            all_labels.append(labels.numpy())
            all_preds.append(preds.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_raw_preds = np.concatenate(all_raw_preds, axis=0)
    
    loss = total_loss / len(data_loader)
    auc = roc_auc_score(all_labels, all_raw_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_raw_preds)
    report = classification_report(all_labels, all_preds, target_names=['injection', 'noise'])
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)


    eval_out = {
        'loss': loss,
        'auc': auc,
        'f1': f1,
        'accuracy': accuracy,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_raw_preds': all_raw_preds,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
    }
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_out

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, results_path, checkpoint_path):
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=10)
    
    train_losses = []
    val_losses = []
    val_aucs = []
    train_times = []
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        train_loss = 0.0
        for input, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()
            
            input = input.to(device)
            labels = labels.squeeze() # Convert labels into
            labels= labels.to(device)
            outputs = model(input)
            #print("outputs: ", outputs)
            labels = labels.view(-1, 1).float().to(device)
            #print("labels output: ", labels )
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        train_times.append(epoch_time)
        
        val_eval_out = evaluate(model, val_loader, device, criterion)
        val_loss = val_eval_out['loss']
        val_auc = val_eval_out['auc']
        
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            # Save evaluation results for the best model
            eval_output_path = os.path.join(results_path, 'best_valid_evaluation_output.json')
            with open(eval_output_path, 'w') as f:
                json_eval = {
                    'loss':  val_eval_out['loss'],
                    'auc': val_eval_out['auc'],
                    'f1': val_eval_out['f1'],
                    'accuracy': val_eval_out['accuracy'],
                    'report': val_eval_out['report'],
                    'fpr': val_eval_out['fpr'].tolist(),
                    'tpr': val_eval_out['tpr'].tolist(),
                }
                json.dump(json_eval, f, indent=4)
            
           
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate on training set
    train_eval_out = evaluate(model, train_loader, device, criterion)
    train_output_path = os.path.join(results_path, 'best_train_evaluation_output.json')
    print("start json")
    with open(train_output_path, 'w') as f:
        json_eval = {
            'loss':  train_eval_out['loss'],
            'auc': train_eval_out['auc'],
            'f1': train_eval_out['f1'],
            'accuracy': train_eval_out['accuracy'],
            'report': train_eval_out['report'],
            'fpr': train_eval_out['fpr'].tolist(),
            'tpr': train_eval_out['tpr'].tolist(),
        }
        json.dump(json_eval, f, indent=4)
    print("finish train json")
    return train_losses, val_losses, val_aucs, train_times