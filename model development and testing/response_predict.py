import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import random
import logging
from pathlib import Path
import numpy as np
from datasets import Dataset

class ChatbotPredictor:
    def __init__(self, model_dir):
        """
        ChatbotPredictor sınıfını başlatır
        Args:
            model_dir: Eğitilmiş model ve gerekli dosyaların bulunduğu dizin
        """
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512  # Model eğitimindeki max_length ile aynı olmalı
        self.confidence_threshold = 0.1  # Güven eşiği
        self.setup_logging()
        self.load_model_and_tokenizer()
        self.load_intent_labels()
        self.load_responses()
        self.feedback_data = {
            'texts': [],
            'labels': [],
            'correct_intents': []
        }
        self.feedback_threshold = 50  # Bu sayıda feedback toplandığında model güncellenecek

    def setup_logging(self):
        """Logging ayarlarını yapılandırır"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prediction.log'),
                logging.StreamHandler()
            ]
        )

    def load_model_and_tokenizer(self):
        """Model ve tokenizer'ı yükler"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model.eval()  # Değerlendirme moduna al
            logging.info(f"Model yüklendi: {self.model_dir}")
        except Exception as e:
            logging.error(f"Model ve tokenizer yüklenirken hata: {e}")
            raise

    def load_intent_labels(self):
        """Intent etiketlerini yükler"""
        try:
            intent_file = self.model_dir / 'intent_labels.json'
            with open(intent_file, 'r', encoding='utf-8') as f:
                self.intent_labels = json.load(f)
            logging.info(f"Intent'ler yüklendi: {len(self.intent_labels)} adet")
        except Exception as e:
            logging.error(f"Intent'ler yüklenirken hata: {e}")
            raise

    def load_responses(self):
        """Yanıtları chatbot_data.json dosyasından yükler"""
        try:
            responses_file = self.model_dir / 'responses.json'
            with open(responses_file, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            logging.info("Yanıtlar yüklendi")
        except Exception as e:
            logging.error(f"Yanıtlar yüklenirken hata: {e}")
            raise

    def preprocess_text(self, text):
        """Metni ön işlemden geçirir"""
        # Büyük harfe çevir (model eğitiminde kullanılan formata uygun olarak)
        text = text.upper()
        # Gereksiz boşlukları temizle
        text = ' '.join(text.split())
        return text

    def get_top_intents(self, probabilities, top_k=3):
        """En yüksek olasılıklı top_k intent'i döndürür"""
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.intent_labels)))
        return [
            (self.intent_labels[idx], prob.item())
            for idx, prob in zip(top_indices.cpu().numpy(), top_probs)
        ]

    def predict(self, text):
        """
        Verilen metin için intent tahmininde bulunur
        Args:
            text: Tahmin edilecek metin
        Returns:
            predicted_intent: Tahmin edilen intent
        """
        try:
            # Metni ön işle
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                
                # En yüksek olasılıklı 3 intent'i al
                top_intents = self.get_top_intents(probabilities)
                
                # En yüksek olasılıklı intent'in güven değeri eşik değerinin üstündeyse
                if top_intents[0][1] >= self.confidence_threshold:
                    return top_intents[0][0]
                else:
                    logging.info(f"Düşük güven: {top_intents}")
                    return None

        except Exception as e:
            logging.error(f"Tahmin hatası: {e}")
            return None

    def get_response(self, intent):
        """
        Verilen intent için rastgele bir yanıt seçer
        Args:
            intent: Yanıt seçilecek intent
        Returns:
            str: Seçilen yanıt
        """
        try:
            available_responses = self.responses.get(intent, ["Üzgünüm, bu konuda yardımcı olamıyorum."])
            
            # Eğer birden fazla yanıt varsa, rastgele seç
            if len(available_responses) > 1:
                return random.choice(available_responses)
            return available_responses[0]
            
        except Exception as e:
            logging.error(f"Yanıt seçme hatası: {e}")
            return "Üzgünüm, bir hata oluştu."

    def collect_feedback(self, text, predicted_intent, correct_intent):
        """Kullanıcı geri bildirimini toplar"""
        try:
            self.feedback_data['texts'].append(text)
            self.feedback_data['labels'].append(
                self.intent_labels.index(correct_intent) if correct_intent else -1
            )
            self.feedback_data['correct_intents'].append(correct_intent)
            
            logging.info(f"Feedback kaydedildi: {text} -> {correct_intent}")
            
            # Yeterli feedback toplandıysa modeli güncelle
            if len(self.feedback_data['texts']) >= self.feedback_threshold:
                self.update_model()
                
        except Exception as e:
            logging.error(f"Feedback kaydedilirken hata: {e}")

    def update_model(self):
        """Toplanan feedback'lerle modeli günceller"""
        try:
            # Sadece doğru etiketli feedback'leri al
            valid_indices = [i for i, label in enumerate(self.feedback_data['labels']) if label != -1]
            
            if not valid_indices:
                logging.info("Güncelleme için yeterli geçerli feedback yok")
                return

            texts = [self.feedback_data['texts'][i] for i in valid_indices]
            labels = [self.feedback_data['labels'][i] for i in valid_indices]

            # Dataset oluştur
            train_dataset = Dataset.from_dict({
                'text': texts,
                'labels': labels
            })

            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None
                )

            train_dataset = train_dataset.map(tokenize_function, batched=True)

            # Eğitim parametreleri
            training_args = TrainingArguments(
                output_dir="feedback_results",
                learning_rate=1e-5,  # Daha düşük learning rate
                per_device_train_batch_size=4,
                num_train_epochs=3,
                save_strategy="no",
                logging_dir='logs',
                logging_steps=1,
            )

            # Trainer oluştur ve modeli güncelle
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset
            )

            trainer.train()
            
            # Feedback verilerini temizle
            self.feedback_data = {'texts': [], 'labels': [], 'correct_intents': []}
            
            logging.info("Model feedback ile güncellendi")
            
        except Exception as e:
            logging.error(f"Model güncellenirken hata: {e}")

def main():
    # Chatbot'u başlat
    try:
        predictor = ChatbotPredictor("saved_model")
        print("\n=== KOSKİ Chatbot Başlatıldı ===")
        print("Çıkmak için 'quit' yazın")
        print("Yanlış cevap için 'düzelt' yazın")
        print("--------------------------------")
        
        while True:
            # Kullanıcı girdisini al
            user_input = input("\nSiz: ").strip()
            
            # Çıkış kontrolü
            if user_input.lower() in ['quit', 'çıkış', 'kapat']:
                print("Chatbot: Görüşmek üzere! İyi günler.")
                break
            
            # Boş girdi kontrolü
            if not user_input:
                print("Chatbot: Lütfen bir soru sorun.")
                continue
            
            # Son soruyu ve tahmini sakla
            last_input = user_input
            intent = predictor.predict(user_input)

            if intent:
                response = predictor.get_response(intent)
                print(f"Chatbot: {response}")
                
                # Kullanıcıya cevabın doğruluğunu sor
                feedback = input("Cevap yardımcı oldu mu? (evet/hayır/düzelt): ").strip().lower()
                
                if feedback == 'düzelt':
                    print("\nMevcut intent kategorileri:")
                    for idx, intent_name in enumerate(predictor.intent_labels):
                        print(f"{idx + 1}. {intent_name}")
                    
                    try:
                        correct_idx = int(input("\nDoğru kategori numarasını girin: ")) - 1
                        if 0 <= correct_idx < len(predictor.intent_labels):
                            correct_intent = predictor.intent_labels[correct_idx]
                            predictor.collect_feedback(last_input, intent, correct_intent)
                            print(f"Feedback kaydedildi. Doğru kategori: {correct_intent}")
                        else:
                            print("Geçersiz kategori numarası")
                    except ValueError:
                        print("Geçersiz giriş")
                
                elif feedback == 'hayır':
                    predictor.collect_feedback(last_input, intent, None)
            else:
                print("Chatbot: Üzgünüm, sorunuzu tam anlayamadım. Lütfen farklı bir şekilde sormayı deneyin.")
                print("Örnek: 'Su faturamı nasıl ödeyebilirim?' veya 'Su kesintisi var mı?'")

    except Exception as e:
        logging.error(f"Program hatası: {e}")
        print("\nBir hata oluştu. Lütfen tekrar deneyin.")

if __name__ == "__main__":
    main()