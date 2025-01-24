from flask import Flask, request, jsonify
import requests
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from pathlib import Path
import json

app = Flask(__name__)

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# OpenWeatherMap API anahtarı
WEATHER_API_KEY = "7549d0b1ceff6a105a44074f3df577e9"
TURKEY_CITIES = [
    "adana", "adıyaman", "afyonkarahisar", "ağrı", "amasya", "ankara", "antalya", "artvin", 
    "aydın", "balıkesir", "bilecik", "bingöl", "bitlis", "bolu", "burdur", "bursa", "çanakkale", 
    "çankırı", "çorum", "denizli", "diyarbakır", "edirne", "elazığ", "erzincan", "erzurum", 
    "eskişehir", "gaziantep", "giresun", "gümüşhane", "hakkari", "hatay", "ısparta", "mersin", 
    "istanbul", "izmir", "kars", "kastamonu", "kayseri", "kırklareli", "kırşehir", "kocaeli", 
    "konya", "kütahya", "malatya", "manisa", "kahramanmaraş", "mardin", "muğla", "muş", "nevşehir", 
    "niğde", "ordu", "rize", "sakarya", "samsun", "siirt", "sinop", "sivas", "tekirdağ", "tokat", 
    "trabzon", "tunceli", "şanlıurfa", "uşak", "van", "yozgat", "zonguldak", "aksaray", "bayburt", 
    "karaman", "kırıkkale", "batman", "şırnak", "bartın", "ardahan", "ığdır", "yalova", "karabük", 
    "kilis", "osmaniye", "düzce"
]

class ChatbotManager:
    def __init__(self):
        self.model_dir = "saved_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.confidence_threshold = 0.1
        self.setup_model()
        
    def setup_model(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model.eval()
            
            # Intent etiketlerini yükle
            with open(Path(self.model_dir) / 'intent_labels.json', 'r', encoding='utf-8') as f:
                self.intent_labels = json.load(f)
                
            # Yanıtları yükle
            with open(Path(self.model_dir) / 'responses.json', 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
                
            logging.info("Chatbot modeli başarıyla yüklendi")
        except Exception as e:
            logging.error(f"Chatbot modeli yüklenirken hata: {e}")
            raise
            
    def predict(self, text):
        try:
            # Metni ön işle
            processed_text = text.upper()
            
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
                confidence, predicted_class = torch.max(probabilities, dim=0)
                
                if confidence.item() >= self.confidence_threshold:
                    intent = self.intent_labels[predicted_class.item()]
                    response = self.get_response(intent)
                    return response, intent, confidence.item()
                    
                return None, None, confidence.item()
                
        except Exception as e:
            logging.error(f"Tahmin hatası: {e}")
            return None, None, 0.0
            
    def get_response(self, intent):
        try:
            available_responses = self.responses.get(intent, ["Üzgünüm, bu konuda yardımcı olamıyorum."])
            return available_responses[0]  # Şimdilik ilk yanıtı döndür
        except Exception as e:
            logging.error(f"Yanıt seçme hatası: {e}")
            return "Bir hata oluştu."

# Global chatbot manager
chatbot = ChatbotManager()

def get_weather(city):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=tr"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        weather_data = response.json()
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        return f"{city.capitalize()} için hava durumu:\nSıcaklık: {temp}°C\nHissedilen: {feels_like}°C\nNem: %{humidity}\nDurum: {description}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Hava durumu API hatası: {e}")
        return "Hava durumu bilgisi alınamadı."
    except KeyError as e:
        logging.error(f"Hava durumu veri hatası: {e}")
        return "Hava durumu verisi işlenirken hata oluştu."


def get_subscriber_info(subscriber_id):
    # Fake abone veritabanı
    fake_data = {
        "12345": {"name": "Ahmet Yılmaz", "status": "aktif", "plan": "premium"},
        "67890": {"name": "Ayşe Demir", "status": "aktif", "plan": "standart"}
    }
    return fake_data.get(subscriber_id, {"error": "Abone bulunamadı"})

def get_current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        original_message = data.get('message', '').strip()
        user_message = original_message.lower()  # Tek bir kez lower case yap
        
        # Hava durumu ve saat sorgularını önce kontrol et
        if "hava" in user_message or "hava durumu" in user_message:
            found_city = None
            for city in TURKEY_CITIES:
                if city in user_message:
                    found_city = city
                    break
            
            if found_city:
                weather_info = get_weather(found_city)
                return jsonify({"response": weather_info})
            
            return jsonify({
                "response": "Hangi şehir için hava durumu bilgisi istersiniz?",
                "expecting": "city"
            })
        
        if "saat" in user_message:
            current_time = get_current_time()
            return jsonify({"response": f"Şu anki saat: {current_time}"})
        
        # Abone numarası kontrolü
        if "abone" in user_message or "endeks" in user_message:
            subscriber_id = ''.join(filter(str.isdigit, user_message))
            if subscriber_id:
                subscriber_info = get_subscriber_info(subscriber_id)
                response = f"Abone Bilgileri:\nİsim: {subscriber_info.get('name', 'Bulunamadı')}\nDurum: {subscriber_info.get('status', 'Bulunamadı')}\nPlan: {subscriber_info.get('plan', 'Bulunamadı')}"
                return jsonify({"response": response})
        
        # Chatbot modeline yönlendir
        response, intent, confidence = chatbot.predict(original_message.upper())
        if response and confidence >= chatbot.confidence_threshold:
            return jsonify({
                "response": response,
                "intent": intent,
                "confidence": confidence
            })
        
        # Chatbot da anlayamazsa
        return jsonify({
            "response": "Üzgünüm, bu mesajı anlayamadım. Su kesintisi, fatura, abonelik gibi konularda yardımcı olabilirim. Ayrıca hava durumu ve saat bilgisi de sorabilirsiniz.",
            "confidence": confidence
        })
        
    except Exception as e:
        logging.error(f"API hatası: {e}")
        return jsonify({
            "error": "Bir hata oluştu",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)