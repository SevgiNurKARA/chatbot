import os
import json
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataset(data):
    texts = []
    labels = []
    for intent in data['intents']:
        for example in intent['examples']:
            texts.append(example)
            labels.append(intent['intent'])
    return texts, labels

def create_label_mapping(unique_labels):
    return {label: i for i, label in enumerate(sorted(unique_labels))}

def main():
    # Ensure output directories exist
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./fine_tuned_model', exist_ok=True)

    # Load the existing model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained('./saved_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained('./saved_model')

    # Load new data
    new_data = load_data('expanded_chatbot_data.json')
    texts, labels = prepare_dataset(new_data)

    # Get all unique labels from your dataset
    unique_labels = set(labels)

    # Create a mapping from your labels to integer indices
    label_mapping = create_label_mapping(unique_labels)
    logger.info(f"Label Mapping: {label_mapping}")

    # Use the mapping to convert your labels to integers
    encoded_labels = [label_mapping[label] for label in labels]

    # Split the data
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

    # Tokenize and encode the data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # Create PyTorch datasets
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    eval_dataset = TensorDataset(eval_encodings['input_ids'], eval_encodings['attention_mask'], torch.tensor(eval_labels))

    # Update the model's classification head if necessary
    if len(unique_labels) != model.config.num_labels:
        model.config.num_labels = len(unique_labels)
        model.classifier = torch.nn.Linear(model.config.hidden_size, len(unique_labels))

    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    batch_size = 16

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss += outputs.loss.item()
        
        avg_eval_loss = eval_loss / len(eval_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Evaluation loss: {avg_eval_loss:.4f}")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

    # Save the label mapping
    with open('./fine_tuned_model/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)

    logger.info("Fine-tuning completed. Model and label mapping saved.")

if __name__ == "__main__":
    main()