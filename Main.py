import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np

# Step 2: Load and Preprocess
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')
true_df['label'] = 0
fake_df['label'] = 1
data = pd.concat([true_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)
data = data[['text', 'label']]

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    data['text'], data['label'], test_size=0.3, random_state=42, stratify=data['label']
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Step 3: Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, labels, max_length=128):
    encodings = tokenizer(
        texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt'
    )
    return TensorDataset(
        encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels.values)
    )

train_dataset = encode_texts(train_texts, train_labels)
val_dataset = encode_texts(val_texts, val_labels)
test_dataset = encode_texts(test_texts, test_labels)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16)

# Step 4: Model and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device) # type: ignore
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss()  

# Step 5: Train
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = [item.to(device) for item in batch]
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = [item.to(device) for item in batch]
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_true, val_preds)
    print(f'Validation Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f}')

# Step 6: Evaluate
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = [item.to(device) for item in batch]
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

print('Test Accuracy:', accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))

# Step 7: Save and Inference
model.save_pretrained('fake_news_model')
tokenizer.save_pretrained('fake_news_model')

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return 'Fake' if torch.argmax(outputs.logits).item() == 1 else 'Real'

sample_text = "Breaking: Aliens invade Earth!"
print(f'Prediction for "{sample_text}": {predict(sample_text)}')