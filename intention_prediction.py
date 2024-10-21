import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_path):
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model ve tokenizer yüklenirken hata: {e}")
        return None, None

def load_label_encoder(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = [intent['intent'] for intent in data['intents']]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder
    except Exception as e:
        logging.error(f"Etiket kodlayıcı yüklenirken hata: {e}")
        return None

def predict(model, tokenizer, label_encoder, text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_intent = label_encoder.inverse_transform([predicted_class])[0]
        return predicted_intent
    except Exception as e:
        logging.error(f"Tahmin yapılırken hata: {e}")
        return None

def main():
    model_path = './saved_model'
    data_file_path = 'chatbot_data.json'  # Orijinal veri dosyasının yolu

    model, tokenizer = load_model_and_tokenizer(model_path)
    label_encoder = load_label_encoder(data_file_path)

    if model is None or tokenizer is None or label_encoder is None:
        return

    # Test örnekleri
    test_texts = [
        "Aboneliğimi nasıl kapatabilirim?",
        "Su sayacım çalışmıyor."
    ]

    for text in test_texts:
        predicted_intent = predict(model, tokenizer, label_encoder, text)
        if predicted_intent:
            logging.info(f"Girdi: '{text}' | Tahmin edilen niyet: {predicted_intent}")
        else:
            logging.warning(f"'{text}' için tahmin yapılamadı.")

if __name__ == "__main__":
    main()