import json
import os
import shutil
import torch
from pathlib import Path
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Define base directory and create Path objects for all directories
BASE_DIR = Path.cwd()
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'
MODEL_DIR = BASE_DIR / 'saved_model'

def setup_directories():
    """Set up directories with proper error handling"""
    directories = [RESULTS_DIR, LOGS_DIR, MODEL_DIR]
    for directory in directories:
        try:
            # Remove directory if it exists
            if directory.exists():
                if directory.is_file():
                    directory.unlink()  # Delete if it's a file
                else:
                    shutil.rmtree(str(directory))  # Delete if it's a directory
            
            # Create directory and all parent directories
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Successfully created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")
            raise

print("Setting up directories...")
setup_directories()

# Load and prepare data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON dosyası geçersiz format: {file_path}")

print("Loading data...")
train_data = load_data('train_data.json')
test_data = load_data('test_data.json')

def prepare_data(data):
    examples = []
    labels = []
    for item in data:
        for example in item['examples']:
            examples.append(example)
            labels.append(item['intent'])
    return examples, labels

print("Preparing datasets...")
train_texts, train_labels = prepare_data(train_data)
test_texts, test_labels = prepare_data(test_data)

# Convert labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Save label encoder classes
label_classes_file = MODEL_DIR / 'label_classes.json'
with label_classes_file.open('w', encoding='utf-8') as f:
    json.dump(list(label_encoder.classes_), f, ensure_ascii=False, indent=2)

print("Loading tokenizer...")
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
except Exception as e:
    raise Exception(f"Tokenizer yüklenirken hata oluştu: {str(e)}")

print("Tokenizing texts...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Create datasets
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'label': train_labels_encoded
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'label': test_labels_encoded
})

print("Loading model...")
num_labels = len(label_encoder.classes_)
try:
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
except Exception as e:
    raise Exception(f"Model yüklenirken hata oluştu: {str(e)}")

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=str(RESULTS_DIR),
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    learning_rate=2e-5,  # Daha düşük learning rate deneyelim
    warmup_steps=500,    # Warm-up ekleyelim
    load_best_model_at_end=True,
    logging_dir=str(LOGS_DIR),
    logging_steps=100,
    save_total_limit=2,
    overwrite_output_dir=True,
    no_cuda=not torch.cuda.is_available(),
    report_to="none"  # Disable wandb and other reporting
)

print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("Starting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Training error details: {str(e)}")
    raise Exception(f"Model eğitimi sırasında hata oluştu: {str(e)}")

print("Evaluating model...")
try:
    eval_results = trainer.evaluate()
    print(f"Değerlendirme sonuçları: {eval_results}")
    
    # Save evaluation results
    eval_results_file = RESULTS_DIR / 'eval_results.json'
    with eval_results_file.open('w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
except Exception as e:
    print(f"Değerlendirme sırasında hata oluştu: {str(e)}")

print("Saving model and tokenizer...")
try:
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("Model ve tokenizer başarıyla kaydedildi.")
except Exception as e:
    print(f"Model kaydedilirken hata oluştu: {str(e)}")