from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import json

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

def main():
    # Önceden eğitilmiş modeli ve tokenizer'ı yükle
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Yeni veriyi yükle
    new_data = load_data('expanded_chatbot_data.json')
    texts, labels = prepare_dataset(new_data)

    # Veriyi eğitim ve değerlendirme setlerine ayır
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2)

    # Veriyi tokenize et
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)

    # Label'ları sayısallaştır
    label_dict = {label: i for i, label in enumerate(set(labels))}
    train_labels = [label_dict[label] for label in train_labels]
    eval_labels = [label_dict[label] for label in eval_labels]

    # Dataset'leri oluştur
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    eval_dataset = Dataset.from_dict({
        'input_ids': eval_encodings['input_ids'],
        'attention_mask': eval_encodings['attention_mask'],
        'labels': eval_labels
    })

    # Model konfigürasyonunu güncelle
    model.config.num_labels = len(label_dict)
    model.classifier = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)

    # Eğitim argümanlarını ayarla
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
    )

    # Trainer oluştur ve eğit
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    # Yeni modeli kaydet
    model.save_pretrained('./new_model')
    tokenizer.save_pretrained('./new_model')

if __name__ == "__main__":
    main()