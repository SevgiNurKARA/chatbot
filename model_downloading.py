import json
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    EarlyStoppingCallback
)

# Model konfigürasyonu için dataclass
@dataclass
class ModelConfig:
    """Model konfigürasyon parametreleri"""
    model_name: str = "dbmdz/distilbert-base-turkish-cased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 100
    early_stopping_patience: int = 3
    test_size: float = 0.2
    min_samples_per_class: int = 2

class ChatbotModelTrainer:
    def __init__(self, config: ModelConfig, base_dir: Path):
        self.config = config
        self.base_dir = Path("C:/Users/sevgi/Desktop/chatbot/")
        self.setup_logging()
        self.setup_directories()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_logging(self) -> None:
        """Logging ayarlarını yapılandırır"""
        try:
            # Log dosyası için dizini oluştur
            log_file = self.base_dir / "training.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            log_format = '%(asctime)s - %(levelname)s - %(message)s'
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        except Exception as e:
            print(f"Logging setup error: {e}")
            raise

    def setup_directories(self) -> None:
        """Gerekli dizinleri oluşturur"""
        try:
            directories = {
                'results': self.base_dir / 'results',
                'logs': self.base_dir / 'logs',
                'saved_model': self.base_dir / 'saved_model'
            }
 
            for dir_path in directories.values():
                # Eğer path varsa ve dosya ise, sil
                if dir_path.exists():
                    if dir_path.is_file():
                        dir_path.unlink()
                    elif dir_path.is_dir():
                        # Dizin varsa içeriğini temizle
                        import shutil
                        shutil.rmtree(dir_path)
                    
                # Dizini oluştur
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {dir_path}")
                
            self.directories = directories
        except Exception as e:
            logging.error(f"Directory setup error: {e}")
            raise

    def load_data(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Eğitim verilerini yükler ve doğrular"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            data = json.loads(file_path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or 'intents' not in data:
                raise ValueError("Invalid data format: Missing 'intents' key")

            texts, labels = [], []
            for intent in data['intents']:
                if all(key in intent for key in ['intent', 'examples']):
                    texts.extend(ex.strip() for ex in intent['examples'] if isinstance(ex, str) and ex.strip())
                    labels.extend([intent['intent']] * len(intent['examples']))

            if not texts or not labels:
                raise ValueError("No valid examples found in dataset")

            logging.info(f"Loaded {len(texts)} examples with {len(set(labels))} unique labels")
            return texts, labels

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[Dataset, Dataset, LabelEncoder]:
        """Veriyi eğitim için hazırlar"""
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Test seti boyutunu hesapla
        n_classes = len(set(labels))
        min_test_size = max(
            self.config.test_size,
            (n_classes * self.config.min_samples_per_class) / len(labels)
        )

        # Veriyi böl
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            encoded_labels,
            test_size=min_test_size,
            stratify=encoded_labels,
            random_state=42
        )

        # Tokenizer'ı yükle
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Dataset'leri oluştur
        train_dataset = self._create_dataset(train_texts, train_labels, tokenizer)
        test_dataset = self._create_dataset(test_texts, test_labels, tokenizer)

        return train_dataset, test_dataset, label_encoder

    def _create_dataset(self, texts: List[str], labels: np.ndarray, tokenizer) -> Dataset:
        """Dataset oluşturur"""
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        return Dataset.from_dict({
            'input_ids': encodings['input_ids'].numpy(),
            'attention_mask': encodings['attention_mask'].numpy(),
            'labels': labels
        })

    def train(self, train_dataset: Dataset, test_dataset: Dataset, num_labels: int) -> Tuple[Any, Dict]:
        """Modeli eğitir"""
        try:
            model = DistilBertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_labels
            ).to(self.device)

            # Dizinlerin var olduğundan emin ol
            for dir_path in self.directories.values():
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(self.directories['results']),
                logging_dir=str(self.directories['logs']),
                evaluation_strategy=IntervalStrategy.STEPS,
                save_strategy=IntervalStrategy.STEPS,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                num_train_epochs=self.config.epochs,
                weight_decay=self.config.weight_decay,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
            )

            trainer.train()
            eval_results = trainer.evaluate()
            return model, eval_results
        
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise

    def save_model(self, model: Any, label_encoder: LabelEncoder) -> None:
        """Model ve ilgili dosyaları kaydeder"""
        try:
            save_dir = self.directories['saved_model']
            
            # Model ve tokenizer'ı kaydet
            model.save_pretrained(save_dir)
            AutoTokenizer.from_pretrained(self.config.model_name).save_pretrained(save_dir)
            
            # Label mapping'i kaydet
            label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
            mapping_file = save_dir / 'label_mapping.json'
            mapping_file.write_text(
                json.dumps(label_mapping, ensure_ascii=False, indent=4),
                encoding='utf-8'
            )
            
            logging.info(f"Model saved successfully to {save_dir}")
            
        except Exception as e:
            logging.error(f"Model saving failed: {e}")
            raise

def main():
    # Konfigürasyon
    config = ModelConfig()
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_file = base_dir / 'chatbot_data.json'

    try:
        # Trainer'ı başlat
        trainer = ChatbotModelTrainer(config, base_dir)
        
        # Veriyi yükle ve hazırla
        texts, labels = trainer.load_data(data_file)
        train_dataset, test_dataset, label_encoder = trainer.prepare_data(texts, labels)
        
        # Modeli eğit
        model, eval_results = trainer.train(train_dataset, test_dataset, len(set(labels)))
        logging.info(f"Evaluation results: {eval_results}")
        
        # Modeli kaydet
        trainer.save_model(model, label_encoder)
        logging.info("Training completed and model saved successfully")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.exception("Detailed error trace:")
        raise

if __name__ == "__main__":
    main()