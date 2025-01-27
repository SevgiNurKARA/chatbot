from flask import Flask, request, jsonify
import requests
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from pathlib import Path
import json
import uuid
import xml.etree.ElementTree as ET

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
# Exchange Rate API
EXCHANGE_API_KEY = "3c8fe66a0c5c0f1b4d8b2f1e"
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

class UserMemory:
    def __init__(self):
        self.memory = {}
        self.chat_history = []
        self.session_id = None
        self.session_dir = None
        self.create_new_session()
    
    def create_new_session(self):
        """Yeni bir session oluştur"""
        self.session_id = str(uuid.uuid4())[:8]
        # Session için yeni memory oluştur
        self.memory = {
            "last_city": None,
            "context": {},
            "last_interaction": None
        }
        self.chat_history = []
        
        # Session için log dizinleri oluştur
        self.session_dir = Path("sessions") / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Session için özel log dosyası ayarla
        self.setup_session_logging()
        
        logging.info(f"Yeni oturum başlatıldı. Session ID: {self.session_id}")
        return self.session_id
    
    def setup_session_logging(self):
        """Her session için özel log dosyası ve formatı ayarla"""
        # Ana log handler'ları temizle
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Session dizininde log dosyaları için klasör oluştur
        log_dir = self.session_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Genel session log dosyası
        session_log = log_dir / "session.log"
        
        # Log formatları
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler - session.log için
        file_handler = logging.FileHandler(session_log, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Root logger'a handler'ları ekle
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
    
    def add_to_chat_history(self, message, response, intent=None):
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "message": message,
            "response": response,
            "intent": intent
        }
        
        self.chat_history.append(chat_entry)
        self._log_chat_entry(chat_entry)
        
        # Konuşmayı log dosyasına da yaz
        logging.info(
            f"Chat | Session: {self.session_id} | Intent: {intent or 'unknown'} | "
            f"Message: {message} | Response: {response}"
        )
    
    def _log_chat_entry(self, chat_entry):
        try:
            # Session bazlı konuşma dosyası
            conversation_file = self.session_dir / "conversation.json"
            
            if conversation_file.exists():
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
            else:
                conversation = {
                    "session_id": self.session_id,
                    "start_time": datetime.now().isoformat(),
                    "messages": []
                }
            
            conversation["messages"].append(chat_entry)
            conversation["last_update"] = datetime.now().isoformat()
            
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"Konuşma kayıt hatası: {e}")
    
    def update_memory(self, key, value):
        self.memory[key] = value
        self.memory["last_interaction"] = datetime.now()
        logging.info(f"Memory güncellendi | Session: {self.session_id} | Key: {key} | Value: {value}")
    
    def get_memory(self):
        return self.memory
    
    def get_chat_history(self):
        return self.chat_history

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
            
            with open(Path(self.model_dir) / 'intent_labels.json', 'r', encoding='utf-8') as f:
                self.intent_labels = json.load(f)
                
            with open(Path(self.model_dir) / 'responses.json', 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
                
            logging.info("Chatbot modeli başarıyla yüklendi")
        except Exception as e:
            logging.error(f"Chatbot modeli yüklenirken hata: {e}")
            raise
            
    def predict(self, text):
        try:
            processed_text = text.upper()
            
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
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
            return available_responses[0]
        except Exception as e:
            logging.error(f"Yanıt seçme hatası: {e}")
            return "Bir hata oluştu."

# Global instances
chatbot = ChatbotManager()
user_memory = UserMemory()

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
    fake_data = {
        "12345": {"name": "Ahmet Yılmaz", "status": "aktif", "plan": "premium"},
        "67890": {"name": "Ayşe Demir", "status": "aktif", "plan": "standart"}
    }
    return fake_data.get(subscriber_id, {"error": "Abone bulunamadı"})

def get_current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def extract_city_from_message(message):
    message = message.lower()
    for city in TURKEY_CITIES:
        if city in message:
            return city
    return None

def get_exchange_rate(currency="USD"):
    """TCMB'den güncel döviz kurlarını çek"""
    base_url = "https://www.tcmb.gov.tr/kurlar/today.xml"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        
        # XML'i parse et
        root = ET.fromstring(response.content)
        
        # Döviz kurlarını bul
        currencies = {
            'USD': {'name': 'ABD DOLARI', 'code': 'USD', 'symbol': '💵'},
            'EUR': {'name': 'EURO', 'code': 'EUR', 'symbol': '💶'},
            'GBP': {'name': 'İNGİLİZ STERLİNİ', 'code': 'GBP', 'symbol': '💷'}
        }
        
        rates = {}
        for currency_data in root.findall('Currency'):
            currency_code = currency_data.get('CurrencyCode')
            if currency_code in ['USD', 'EUR', 'GBP']:
                forex_buying = float(currency_data.find('ForexBuying').text)
                forex_selling = float(currency_data.find('ForexSelling').text)
                name = currencies[currency_code]['name']
                symbol = currencies[currency_code]['symbol']
                rates[currency_code] = {
                    'name': name,
                    'buying': forex_buying,
                    'selling': forex_selling,
                    'symbol': symbol
                }
        
        # Yanıt metni oluştur
        line_width = 40
        separator = "─" * line_width
        
        response_text = "📊 TCMB Güncel Döviz Kurları\n"
        response_text += separator + "\n\n"
        
        for curr_code, rate in rates.items():
            response_text += f"{rate['symbol']} {rate['name']}\n"
            response_text += f"➡️ Alış:  {rate['buying']:.4f} ₺\n"
            response_text += f"⬅️ Satış: {rate['selling']:.4f} ₺\n"
            response_text += separator + "\n\n"
        
        response_text += f"🕒 Son Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
        response_text += f"🏦 Kaynak: T.C. Merkez Bankası"
        
        return response_text
    except requests.exceptions.RequestException as e:
        logging.error(f"TCMB döviz kuru API hatası: {e}")
        return "Döviz kuru bilgisi alınamadı. Lütfen daha sonra tekrar deneyin."
    except (ET.ParseError, KeyError, AttributeError, ValueError) as e:
        logging.error(f"TCMB döviz kuru veri hatası: {e}")
        return "Döviz kuru verisi işlenirken hata oluştu. Lütfen daha sonra tekrar deneyin."

@app.route('/', methods=['GET'])
def get_active_session():
    """Ana endpoint - her istekte yeni bir session ID oluşturur"""
    # Yeni session oluştur
    new_session_id = user_memory.create_new_session()
    
    return jsonify({
        "status": "active",
        "session_id": new_session_id,
        "chat_endpoint": f"/chat/{new_session_id}",
        "start_time": datetime.now().isoformat(),
        "usage": {
            "description": "Bu API ile sohbet etmek için:",
            "endpoint": f"/chat/{new_session_id}",
            "method": "POST",
            "body_format": {
                "message": "mesajınız"
            }
        }
    })

@app.route('/chat/<session_id>', methods=['GET', 'POST'])
def chat_with_mem(session_id):
    # Session ID kontrolü
    if session_id != user_memory.session_id:
        return jsonify({
            "error": "Geçersiz veya süresi dolmuş oturum",
            "current_session": user_memory.session_id,
            "message": "Lütfen ana endpoint'ten ('/') güncel oturum bilgisini alın"
        }), 401

    # GET isteği için endpoint bilgisi döndür
    if request.method == 'GET':
        return jsonify({
            "status": "active",
            "session_id": session_id,
            "message": "Bu endpoint'e POST methodu ile mesaj gönderebilirsiniz",
            "usage": {
                "method": "POST",
                "body_format": {
                    "message": "mesajınız"
                }
            }
        })
    
    # POST isteği işleme
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({
                "error": "Geçersiz istek",
                "message": "Request body'de 'message' alanı gerekli"
            }), 400
            
        original_message = data.get('message', '').strip()
        user_message = original_message.lower()
        
        # Mevcut hafızayı al
        current_memory = user_memory.get_memory()
        
        response_data = None
        intent = None
        
        # Döviz kuru sorgularını kontrol et
        if any(word in user_message for word in ["döviz", "kur", "euro", "dolar", "sterlin", "usd", "eur", "gbp"]):
            exchange_info = get_exchange_rate()
            response_data = {"response": exchange_info}
            intent = "currency_query"
        
        # Hava durumu sorgularını kontrol et
        elif "hava" in user_message or "hava durumu" in user_message:
            # Mesajdan şehir çıkar
            city = extract_city_from_message(user_message)
            
            # Eğer mesajda şehir yoksa, hafızadaki son şehri kontrol et
            if not city:
                last_city = current_memory.get("last_city")
                if last_city:
                    response_data = {
                        "response": f"{last_city.capitalize()} için hava durumunu kontrol ediyorum...\n" + get_weather(last_city)
                    }
                    intent = "weather_query"
                else:
                    # Hava durumu context'ini kaydet
                    user_memory.update_memory("context", {"type": "weather_waiting_city"})
                    response_data = {
                        "response": "Hangi şehir için hava durumu bilgisi istersiniz?",
                        "expecting": "city"
                    }
                    intent = "weather_city_request"
            else:
                # Şehir bulunduysa hafızaya kaydet ve hava durumunu getir
                user_memory.update_memory("last_city", city)
                user_memory.update_memory("context", {"type": "weather_complete"})
                weather_info = get_weather(city)
                response_data = {"response": weather_info}
                intent = "weather_query"
        
        # Sadece şehir ismi yazıldıysa ve context weather_waiting_city ise
        elif city := extract_city_from_message(user_message):
            # Şehir ismi sadece hava durumu sorulduktan sonra işlenmeli
            if current_memory.get("context", {}).get("type") == "weather_waiting_city":
                user_memory.update_memory("last_city", city)
                user_memory.update_memory("context", {"type": "weather_complete"})
                weather_info = get_weather(city)
                response_data = {"response": weather_info}
                intent = "weather_query"
            else:
                response_data = {
                    "response": "Önce hava durumunu sormalısınız. Örneğin: 'Hava durumu nasıl?' şeklinde sorabilirsiniz."
                }
                intent = "invalid_city_query"
        
        # Yarın, bugün gibi zaman belirten kelimeler varsa ve son şehir kayıtlıysa
        elif any(indicator in user_message for indicator in ["yarın", "bugün", "sonra", "akşam", "sabah", "öğlen"]):
            if current_memory.get("context", {}).get("type") == "weather_complete":
                last_city = current_memory.get("last_city")
                if last_city:
                    response_data = {
                        "response": f"{last_city.capitalize()} için hava durumunu kontrol ediyorum...\n" + get_weather(last_city)
                    }
                    intent = "weather_query"
        
        elif "saat" in user_message:
            current_time = get_current_time()
            response_data = {"response": f"Şu anki saat: {current_time}"}
            intent = "time_query"
        
        elif "abone" in user_message or "endeks" in user_message:
            subscriber_id = ''.join(filter(str.isdigit, user_message))
            if subscriber_id:
                subscriber_info = get_subscriber_info(subscriber_id)
                response = f"Abone Bilgileri:\nİsim: {subscriber_info.get('name', 'Bulunamadı')}\nDurum: {subscriber_info.get('status', 'Bulunamadı')}\nPlan: {subscriber_info.get('plan', 'Bulunamadı')}"
                response_data = {"response": response}
                intent = "subscriber_query"
        
        # Eğer özel durumlar karşılanmadıysa chatbot'a yönlendir
        if not response_data:
            response, chatbot_intent, confidence = chatbot.predict(original_message.upper())
            if response and confidence >= chatbot.confidence_threshold:
                response_data = {
                    "response": response,
                    "intent": chatbot_intent,
                    "confidence": confidence
                }
                intent = chatbot_intent
            else:
                # Context'i temizle eğer alakasız bir soru sorulduysa
                user_memory.update_memory("context", {})
                response_data = {
                    "response": "Üzgünüm, bu mesajı anlayamadım. Su kesintisi, fatura, abonelik gibi konularda yardımcı olabilirim. Ayrıca hava durumu ve saat bilgisi de sorabilirsiniz.",
                    "confidence": confidence
                }
                intent = "unknown"
        
        # Sohbeti logla
        user_memory.add_to_chat_history(
            message=original_message,
            response=response_data["response"],
            intent=intent
        )
        
        # Session ID'yi response'a ekle
        response_data["session_id"] = user_memory.session_id
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"API hatası: {e}")
        error_response = {
            "error": "Bir hata oluştu",
            "details": str(e),
            "session_id": user_memory.session_id
        }
        user_memory.add_to_chat_history(
            message=original_message if 'original_message' in locals() else "Unknown message",
            response=str(e),
            intent="error"
        )
        return jsonify(error_response), 500

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    try:
        history = user_memory.get_chat_history()
        return jsonify({
            "session_id": user_memory.session_id,
            "history": history
        })
    except Exception as e:
        logging.error(f"Sohbet geçmişi hatası: {e}")
        return jsonify({
            "error": "Sohbet geçmişi alınamadı",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
