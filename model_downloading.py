import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import logging
from transformers import IntervalStrategy

# Logging ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Veri setini yükleme
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Veri yüklenirken hata: {e}")
        return None

# Dizin oluşturma fonksiyonu
def create_directory(directory):
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"{directory} dizini oluşturuldu veya zaten mevcut.")
    except Exception as e:
        logging.error(f"{directory} dizini oluşturulurken hata: {e}")
        raise

# Ana fonksiyon
def main():
    # Çıktı dizinlerini oluştur
    try:
        create_directory('/results')
        create_directory('/logs')
        create_directory('/saved_model')
    except Exception as e:
        logging.error(f"Dizinler oluşturulurken hata: {e}")
        return

    # Orijinal veri setini yükleme
    original_data = load_data('chatbot_data.json')
    if original_data is None:
        return

    # Veriyi hazırlama
    texts, labels = [], []
    for intent in original_data.get('intents', []):
        for example in intent.get('examples', []):
            texts.append(example)
            labels.append(intent['intent'])

    logging.info(f"Toplam örnek sayısı: {len(texts)}")
    logging.info(f"Benzersiz etiket sayısı: {len(set(labels))}")

    # Veriyi eğitim ve test setlerine ayırma
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    all_labels = train_labels + test_labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

# Etiketleri sayısal değerlere dönüştür
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    # Tokenizer yükleme
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    except Exception as e:
        logging.error(f"Tokenizer yüklenirken hata: {e}")
        return

    # Tokenizasyon
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    # Dataset oluşturma
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels_encoded
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels_encoded
    })

    logging.info(f"Eğitim seti boyutu: {len(train_dataset)}")
    logging.info(f"Test seti boyutu: {len(test_dataset)}")

    # Modeli yükleme
    num_labels = len(label_encoder.classes_)
    try:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    except Exception as e:
        logging.error(f"Model yüklenirken hata: {e}")
        return

    # Eğitim argümanları ayarlama
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="steps",
        save_strategy="steps",
        evaluation_strategy="steps",  # Eski versiyon uyumluluğu için
        eval_steps=100,
        save_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='/logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Trainer oluşturma
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Modeli eğitme
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Model eğitilirken hata: {e}")
        logging.debug("Trainer detayları:", exc_info=True)
        return

    # Modeli değerlendirme
    try:
        eval_results = trainer.evaluate()
        logging.info(f"Değerlendirme sonuçları: {eval_results}")
    except Exception as e:
        logging.error(f"Model değerlendirilirken hata: {e}")
# Label mapping oluşturma ve kaydetme
    def save_label_mapping(label_encoder, directory='./saved_model'):
    # Dizin oluştur
        os.makedirs(directory, exist_ok=True)
    
    # Label mapping oluştur
        label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    
    # Dosyayı kaydet
        try:
            with open(f"{directory}/label_mapping.json", 'w') as f:
                json.dump(label_mapping, f, ensure_ascii=False, indent=4)
            print("Label mapping başarıyla kaydedildi.")
        except Exception as e:
            print(f"Label mapping kaydedilirken hata: {e}")
    save_label_mapping(label_encoder, './saved_model')

    # Modeli ve tokenizer'ı kaydetme
    try:
        model.save_pretrained('./saved_model')
        tokenizer.save_pretrained('./saved_model')
        logging.info("Model ve tokenizer başarıyla kaydedildi.")
    except Exception as e:
        logging.error(f"Model ve tokenizer kaydedilirken hata: {e}")

if __name__ == "__main__":
    main()