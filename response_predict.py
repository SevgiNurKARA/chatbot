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

def load_label_encoder_and_responses(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        intents = data['intents']
        labels = [intent['intent'] for intent in intents]
        responses = {intent['intent']: intent['responses'] for intent in intents}
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder, responses
    except Exception as e:
        logging.error(f"Etiket kodlayıcı ve cevaplar yüklenirken hata: {e}")
        return None, None

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

def get_response(intent, responses):
    import random
    return random.choice(responses.get(intent, ["Üzgünüm, bu konuda bir cevabım yok."]))

def chatbot():
    model_path = './saved_model'
    data_file_path = 'chatbot_data.json'  # Orijinal veri dosyasının yolu

    model, tokenizer = load_model_and_tokenizer(model_path)
    label_encoder, responses = load_label_encoder_and_responses(data_file_path)

    if model is None or tokenizer is None or label_encoder is None or responses is None:
        print("Chatbot başlatılamadı. Lütfen hata mesajlarını kontrol edin.")
        return

    print("Chatbot hazır! Çıkmak için 'quit' yazın.")
    
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Görüşmek üzere!")
            break

        predicted_intent = predict(model, tokenizer, label_encoder, user_input)
        if predicted_intent:
            response = get_response(predicted_intent, responses)
            print(f"Chatbot: {response}")
        else:
            print("Chatbot: Üzgünüm, şu anda cevap veremiyorum. Lütfen tekrar deneyin.")

if __name__ == "__main__":
    chatbot()