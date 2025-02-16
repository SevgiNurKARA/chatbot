import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import logging

class ChatbotTrainer:
    def __init__(self):
        # Doğru model adını kullanıyoruz
        self.model_name = "dbmdz/bert-base-turkish-uncased"  # Türkçe için optimize edilmiş model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512  # Uzun cevaplar için max length arttırıldı
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        
    def load_data(self, file_path):
        """JSON dosyasından veriyi yükler ve veri artırma yapar"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        texts = []
        labels = []
        self.intent_labels = []
        self.responses = {}  # Her intent için yanıtları saklayacağız
        
        # Her intent için örnekleri topla
        for idx, intent in enumerate(data['intents']):
            intent_name = intent['intent']
            self.intent_labels.append(intent_name)
            self.responses[intent_name] = intent.get('responses', [])
            
            # Mevcut örnekleri ekle
            examples = intent['examples']
            texts.extend(examples)
            labels.extend([idx] * len(examples))
            
            # Veri artırma: Benzer örnekleri birleştir
            for i in range(len(examples)):
                for j in range(i + 1, len(examples)):
                    if len(examples[i].split()) + len(examples[j].split()) <= 30:  # Çok uzun cümleler oluşturmamak için
                        combined = examples[i] + " ve " + examples[j]
                        texts.append(combined)
                        labels.append(idx)
                    
        logging.info(f"Toplam örnek sayısı: {len(texts)}")
        logging.info(f"Toplam intent sayısı: {len(self.intent_labels)}")
        
        return texts, labels
    
    def prepare_data(self, texts, labels):
        """Veriyi eğitim için hazırlar"""
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, stratify=labels, random_state=42
        )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors=None  # Batch işleme için None kullanıyoruz
            )
        
        # Dataset'leri oluştur
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        
        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Değerlendirme metriklerini hesaplar"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Doğruluk (accuracy) hesapla
        accuracy = (predictions == labels).mean()
        
        # Her intent için F1 skoru hesapla
        f1_scores = {}
        for idx, intent in enumerate(self.intent_labels):
            true_positives = ((predictions == idx) & (labels == idx)).sum()
            pred_positives = (predictions == idx).sum()
            actual_positives = (labels == idx).sum()
            
            precision = true_positives / pred_positives if pred_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores[f"f1_{intent}"] = f1
        
        metrics = {"accuracy": accuracy, **f1_scores}
        return metrics
    
    def train(self, train_dataset, val_dataset):
        """Modeli eğitir"""
        num_labels = len(self.intent_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Eğitim parametreleri
        training_args = TrainingArguments(
            output_dir="results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=15,  # Epoch sayısı arttırıldı
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            push_to_hub=False,
            logging_dir='logs',
            logging_steps=10,
            warmup_steps=500,  # Warmup eklendi
            fp16=True if torch.cuda.is_available() else False,  # GPU varsa FP16 kullan
        )
        
        # Trainer'ı oluştur
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Eğitimi başlat
        trainer.train()
        
        return model
    
    def save_model(self, model, save_dir):
        """Modeli ve gerekli dosyaları kaydeder"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Modeli kaydet
        model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Intent etiketlerini ve yanıtları kaydet
        with open(os.path.join(save_dir, 'intent_labels.json'), 'w', encoding='utf-8') as f:
            json.dump(self.intent_labels, f, ensure_ascii=False, indent=4)
            
        with open(os.path.join(save_dir, 'responses.json'), 'w', encoding='utf-8') as f:
            json.dump(self.responses, f, ensure_ascii=False, indent=4)
            
        logging.info(f"Model ve gerekli dosyalar kaydedildi: {save_dir}")

def main():
    # Trainer'ı oluştur
    trainer = ChatbotTrainer()
    
    # Veri dosyasının yolu
    data_file = "chatbot_data.json"
    
    # Veriyi yükle
    texts, labels = trainer.load_data(data_file)
    
    # Veriyi hazırla
    train_dataset, val_dataset = trainer.prepare_data(texts, labels)
    
    # Modeli eğit
    model = trainer.train(train_dataset, val_dataset)
    
    # Modeli kaydet
    save_dir = "saved_model"
    trainer.save_model(model, save_dir)
    print("Eğitim tamamlandı ve model kaydedildi!")

if __name__ == "__main__":
    main()