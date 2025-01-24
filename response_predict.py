import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import logging
from pathlib import Path

class ChatbotPredictor:
    def __init__(self, model_dir):
        """
        ChatbotPredictor sınıfını başlatır
        Args:
            model_dir: Eğitilmiş model ve gerekli dosyaların bulunduğu dizin
        """
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 256
        self.setup_logging()
        self.load_model_and_tokenizer()
        self.load_intent_labels()
        self.load_responses()

    def setup_logging(self):
        """Logging ayarlarını yapılandırır"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_model_and_tokenizer(self):
        """Model ve tokenizer'ı yükler"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model.eval()  # Değerlendirme moduna al
            logging.info("Model ve tokenizer başarıyla yüklendi")
        except Exception as e:
            logging.error(f"Model ve tokenizer yüklenirken hata: {e}")
            raise

    def load_intent_labels(self):
        """Intent etiketlerini yükler"""
        try:
            intent_file = self.model_dir / 'intent_labels.json'
            with open(intent_file, 'r', encoding='utf-8') as f:
                self.intent_labels = json.load(f)
            logging.info(f"Intent etiketleri yüklendi: {len(self.intent_labels)} adet")
        except Exception as e:
            logging.error(f"Intent etiketleri yüklenirken hata: {e}")
            raise

    def load_responses(self):
        """Yanıtları chatbot_data.json dosyasından yükler"""
        try:
            with open('chatbot_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.responses = {
                intent['intent']: intent.get('responses', ["Üzgünüm, bu konuda yanıt veremiyorum."])
                for intent in data['intents']
            }
            logging.info("Yanıtlar başarıyla yüklendi")
        except Exception as e:
            logging.error(f"Yanıtlar yüklenirken hata: {e}")
            raise

    def predict(self, text):
        """
        Verilen metin için intent tahmininde bulunur
        Args:
            text: Tahmin edilecek metin
        Returns:
            predicted_intent: Tahmin edilen intent
        """
        try:
            # Metni tokenize et
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

            # Eğer güven değeri düşükse None döndür
            if confidence.item() < 0.5:
                return None

            # Intent'i döndür
            predicted_intent = self.intent_labels[predicted_class.item()]
            return predicted_intent

        except Exception as e:
            logging.error(f"Tahmin yapılırken hata: {e}")
            return None

    def get_response(self, intent):
        """
        Verilen intent için rastgele bir yanıt seçer
        Args:
            intent: Yanıt seçilecek intent
        Returns:
            str: Seçilen yanıt
        """
        responses = self.responses.get(intent, ["Üzgünüm, bu konuda yanıt veremiyorum."])
        return random.choice(responses)

def main():
    # Chatbot'u başlat
    try:
        predictor = ChatbotPredictor("saved_model")
        print("Chatbot başlatıldı! Çıkmak için 'quit' yazın.")
        
        while True:
            # Kullanıcı girdisini al
            user_input = input("\nSiz: ").strip()
            
            # Çıkış kontrolü
            if user_input.lower() == 'quit':
                print("Chatbot: Görüşmek üzere!")
                break
            
            # Boş girdi kontrolü
            if not user_input:
                print("Chatbot: Lütfen bir şeyler yazın.")
                continue
            
            # Tahminde bulun ve yanıt ver
            predicted_intent = predictor.predict(user_input)
            if predicted_intent:
                response = predictor.get_response(predicted_intent)
                print(f"Chatbot: {response}")
            else:
                print("Chatbot: Üzgünüm, sizi anlayamadım. Lütfen tekrar dener misiniz?")

    except Exception as e:
        logging.error(f"Chatbot çalışırken hata oluştu: {e}")
        print("Chatbot başlatılırken bir hata oluştu. Lütfen log dosyasını kontrol edin.")

if __name__ == "__main__":
    main()