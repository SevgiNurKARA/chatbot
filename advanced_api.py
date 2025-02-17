from flask import Flask, request, jsonify, render_template
from datetime import datetime
import logging
from pathlib import Path
import json
import uuid
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xml.etree.ElementTree as ET
import os

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Sahte fatura veritabanı - Yeni yapı
BILL_DATABASE = {
    # TC: (Abone No, Doğum Tarihi): Fatura Bilgileri
    "12345678901": {
        ("123456", "1990-01-01"): {
            "bill_amount": 150.75,
            "water_usage": 15.3,
            "bill_date": "2024-03-01"
        }
    },
    # Abone No: (Doğum Tarihi): Fatura Bilgileri
    "123456": {
        "1990-01-01": {
            "bill_amount": 150.75,
            "water_usage": 15.3,
            "bill_date": "2024-03-01"
        }
    }
}

# Sahte su kesintisi veritabanı
WATER_OUTAGE_DATABASE = {
    "karşıyaka": {
        "status": True,  # Kesinti var
        "start_time": "2024-03-14 10:00:00",
        "end_time": "2024-03-14 18:00:00",
        "reason": "Ana boru hattı bakımı"
    },
    "bornova": {
        "status": False,  # Kesinti yok
        "last_update": "2024-03-14 09:00:00"
    },
    "konak": {
        "status": True,
        "start_time": "2024-03-14 09:00:00",
        "end_time": "2024-03-14 16:00:00",
        "reason": "Acil altyapı çalışması"
    }
}

# Sahte sayaç veritabanı
METER_DATABASE = {
    # Abone No: Sayaç Bilgileri
    "123456": {
        "meter_id": "M123456",
        "last_reading": 1523.45,
        "last_reading_date": "2024-03-01",
        "status": "Aktif",
        "installation_date": "2023-01-15",
        "meter_type": "Dijital",
        "address": "Örnek Mahallesi, 123. Sokak No:45 Bornova/İzmir"
    }
}

# Şikayet veritabanı
COMPLAINTS_DATABASE = {
    "complaints": [],
    "last_complaint_id": 0
}

# Türkiye şehirleri
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
        self.sessions = {}  # Tüm sessionları tutacak sözlük
        self.current_session_id = None
        self.chat_history_dir = Path('chat_histories')  # Konuşma geçmişleri için klasör
        self.chat_history_dir.mkdir(exist_ok=True)  # Klasörü oluştur
    
    def create_new_session(self):
        """Yeni bir session oluştur"""
        try:
            # Benzersiz bir session ID oluştur
            session_id = str(uuid.uuid4())[:8]
            
            # Session ID'nin benzersiz olduğundan emin ol
            while session_id in self.sessions:
                session_id = str(uuid.uuid4())[:8]
            
            # Yeni session oluştur
            self.sessions[session_id] = {
                "context": {},
                "chat_history": [],
                "last_interaction": datetime.now(),
                "created_at": datetime.now(),
                "user_id": f"user_{session_id}"  # Her session için benzersiz kullanıcı ID'si
            }
            
            self.current_session_id = session_id
            logging.info(f"Yeni oturum başlatıldı. Session ID: {session_id}")
            
            return session_id
            
        except Exception as e:
            logging.error(f"Session oluşturma hatası: {e}")
            return None
    
    def add_to_chat_history(self, session_id, message, sender):
        """Konuşma geçmişine mesaj ekle ve dosyaya kaydet"""
        if session_id in self.sessions:
            # Mesajı geçmişe ekle
            chat_history = self.sessions[session_id].get("chat_history", [])
            
            # Son mesajı kontrol et ve tekrarı önle
            if not chat_history or not (chat_history[-1]["message"] == message and chat_history[-1]["sender"] == sender):
                message_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sender": sender,
                    "message": message,
                    "message_id": str(uuid.uuid4())[:8]  # Benzersiz mesaj ID'si
                }
                chat_history.append(message_data)
                self.sessions[session_id]["chat_history"] = chat_history
                
                # Son etkileşim zamanını güncelle
                self.sessions[session_id]["last_interaction"] = datetime.now()
                
                # Konuşma geçmişini dosyaya kaydet
                self._save_chat_history(session_id)
                return True
            return False
        return False
    
    def _save_chat_history(self, session_id):
        """Konuşma geçmişini dosyaya kaydet"""
        try:
            if session_id in self.sessions:
                session_data = self.sessions[session_id]
                user_id = session_data["user_id"]
                
                # Kullanıcıya özel klasör oluştur
                user_dir = self.chat_history_dir / user_id
                user_dir.mkdir(exist_ok=True)
                
                # Konuşma geçmişi dosyası
                history_file = user_dir / f"chat_history_{datetime.now().strftime('%Y%m%d')}.json"
                
                # Mevcut geçmişi yükle veya yeni oluştur
                history_data = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": []
                }
                
                # Mevcut mesaj ID'lerini takip etmek için set oluştur
                existing_message_ids = set()
                
                if history_file.exists():
                    with open(history_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        # Mevcut mesajları ID'lerine göre kontrol et
                        existing_message_ids = {msg.get("message_id") for msg in existing_data.get("messages", [])}
                        history_data["messages"] = existing_data.get("messages", [])
                
                # Yeni mesajları ekle (tekrar kontrolü yaparak)
                for message in session_data["chat_history"]:
                    if "message_id" in message and message["message_id"] not in existing_message_ids:
                        history_data["messages"].append(message)
                        existing_message_ids.add(message["message_id"])
                
                # Dosyaya kaydet
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(history_data, f, ensure_ascii=False, indent=4)
                
                logging.info(f"Konuşma geçmişi kaydedildi: {history_file}")
        except Exception as e:
            logging.error(f"Konuşma geçmişi kaydetme hatası: {e}")
    
    def get_user_chat_history(self, user_id, date=None):
        """Belirli bir kullanıcının konuşma geçmişini getir"""
        try:
            user_dir = self.chat_history_dir / user_id
            if not user_dir.exists():
                return []
            
            # Belirli bir tarih için geçmiş
            if date:
                history_file = user_dir / f"chat_history_{date}.json"
                if history_file.exists():
                    with open(history_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Mesajları benzersiz ID'lere göre filtrele
                        seen_message_ids = set()
                        unique_messages = []
                        for msg in data.get("messages", []):
                            msg_id = msg.get("message_id")
                            if msg_id and msg_id not in seen_message_ids:
                                unique_messages.append(msg)
                                seen_message_ids.add(msg_id)
                        return unique_messages
                return []
            
            # Tüm geçmiş
            all_messages = []
            seen_message_ids = set()
            
            for history_file in sorted(user_dir.glob("chat_history_*.json")):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for msg in data.get("messages", []):
                        msg_id = msg.get("message_id")
                        if msg_id and msg_id not in seen_message_ids:
                            all_messages.append(msg)
                            seen_message_ids.add(msg_id)
            
            return all_messages
            
        except Exception as e:
            logging.error(f"Konuşma geçmişi getirme hatası: {e}")
            return []
    
    def get_session(self, session_id):
        """Belirli bir session'ı getir"""
        session = self.sessions.get(session_id)
        if session:
            session["last_interaction"] = datetime.now()
        return session
    
    def update_session(self, session_id, key, value):
        """Session bilgilerini güncelle"""
        if session_id in self.sessions:
            self.sessions[session_id][key] = value
            self.sessions[session_id]["last_interaction"] = datetime.now()
            logging.info(f"Session güncellendi | Session: {session_id} | Key: {key}")
            return True
        return False
    
    def is_valid_session(self, session_id):
        """Session'ın geçerli olup olmadığını kontrol et"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        last_interaction = session.get("last_interaction")
        
        # 30 dakika içinde etkileşim olmamışsa session'ı geçersiz say
        if last_interaction:
            time_diff = (datetime.now() - last_interaction).total_seconds() / 60
            if time_diff > 30:
                # Son konuşma geçmişini kaydet
                self._save_chat_history(session_id)
                del self.sessions[session_id]
                return False
                
        return True

# Global instance
user_memory = UserMemory()

def validate_tc(tc_no):
    """TC Kimlik numarası doğrulama"""
    if not tc_no.isdigit() or len(tc_no) != 11:
        return False
    return True

def validate_subscriber_no(subscriber_no):
    """Abone numarası doğrulama"""
    if not subscriber_no.isdigit() or len(subscriber_no) != 6:
        return False
    return True

def validate_date(date_str):
    """Tarih formatı doğrulama (YYYY-MM-DD)"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def get_bill_info(identifier, birth_date, subscriber_no=None):
    """Fatura bilgilerini getir"""
    try:
        if len(identifier) == 11:  # TC ile sorgu
            if subscriber_no:
                return BILL_DATABASE.get(identifier, {}).get((subscriber_no, birth_date))
        else:  # Abone no ile sorgu
            return BILL_DATABASE.get(identifier, {}).get(birth_date)
    except Exception as e:
        logging.error(f"Fatura bilgisi getirme hatası: {e}")
        return None

def get_exchange_rate():
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
        response_text = "📊 TCMB Güncel Döviz Kurları\n\n"
        
        for curr_code, rate in rates.items():
            response_text += f"{rate['symbol']} {rate['name']}\n"
            response_text += f"➡ Alış:  {rate['buying']:.4f} ₺\n"
            response_text += f"⬅ Satış: {rate['selling']:.4f} ₺\n\n"
        
        response_text += f"🕒 Son Güncelleme: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
        response_text += f"🏦 Kaynak: T.C. Merkez Bankası"
        
        return response_text
    except Exception as e:
        logging.error(f"Döviz kuru hatası: {e}")
        return "Döviz kuru bilgisi alınamadı."

def get_weather(city):
    """Hava durumu bilgisini getir"""
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

def get_current_time():
    """Güncel saati formatla"""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def check_water_outage(district):
    """Su kesintisi kontrolü"""
    district = district.lower()
    outage_info = WATER_OUTAGE_DATABASE.get(district)
    
    if not outage_info:
        return "Bu mahalle için bilgi bulunamadı."
    
    if outage_info["status"]:
        return (
            f"{district.capitalize()} bölgesinde su kesintisi var.\n"
            f"Başlangıç: {outage_info['start_time']}\n"
            f"Bitiş: {outage_info['end_time']}\n"
            f"Sebep: {outage_info['reason']}"
        )
    else:
        return f"{district.capitalize()} bölgesinde şu anda su kesintisi bulunmamaktadır.\nSon güncelleme: {outage_info['last_update']}"

def save_complaint(complaint_data):
    """Şikayet bilgilerini kaydet"""
    try:
        # Gerekli alanların kontrolü
        required_fields = ["complaint_type", "description", "subscriber_no", "birth_date", "address"]
        if not all(field in complaint_data for field in required_fields):
            logging.error("Eksik şikayet bilgisi")
            return None
            
        # Şikayet ID'sini artır
        COMPLAINTS_DATABASE["last_complaint_id"] += 1
        complaint_id = COMPLAINTS_DATABASE["last_complaint_id"]
        
        # Şikayet bilgilerini hazırla
        complaint = {
            "id": complaint_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "complaint_type": complaint_data["complaint_type"],
            "description": complaint_data["description"],
            "subscriber_no": complaint_data["subscriber_no"],
            "birth_date": complaint_data["birth_date"],
            "address": complaint_data["address"],
            "status": "Yeni",
            "priority": "Normal",
            "notification_sent": True,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Şikayeti kaydet
        COMPLAINTS_DATABASE["complaints"].append(complaint)
        
        # Şikayetleri dosyaya kaydet
        save_complaints_to_file()
        
        logging.info(f"Yeni şikayet kaydedildi. ID: {complaint_id}, Tür: {complaint_data['complaint_type']}")
        return complaint_id
        
    except Exception as e:
        logging.error(f"Şikayet kaydetme hatası: {e}")
        return None

def save_complaints_to_file():
    """Şikayetleri JSON dosyasına kaydet"""
    try:
        complaints_file = Path('complaints.json')
        with open(complaints_file, 'w', encoding='utf-8') as f:
            json.dump(COMPLAINTS_DATABASE, f, ensure_ascii=False, indent=4)
        logging.info("Şikayetler dosyaya kaydedildi")
    except Exception as e:
        logging.error(f"Şikayet dosyası kaydetme hatası: {e}")
        raise  # Hatayı yukarı ilet

def load_complaints_from_json(file_path='complaint.json'):
    """
    Load complaints from a JSON file.
    
    :param file_path: Path to the JSON file containing complaints.
    :return: List of complaints.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Erişilecek anahtar: "complaints"
    complaints = data.get("complaints", [])
    return complaints

def get_complaints_for_dashboard():
    """
    Retrieve complaints to be displayed on the dashboard.
    
    :return: List of complaints formatted for the dashboard.
    """
    complaints = load_complaints_from_json('complaint.json')
    formatted_complaints = format_complaints_for_dashboard(complaints)
    return formatted_complaints

def format_complaints_for_dashboard(complaints):
    """
    Format complaints for display on the dashboard.
    
    :param complaints: List of complaints.
    :return: Formatted list of complaints.
    """
    return [{"id": c["id"], "message": c["description"], "timestamp": c["timestamp"], "subscriber_no": c["subscriber_no"], "address": c["address"], "complaint_type": c["complaint_type"], "status": c["status"]} for c in complaints]

@app.route('/api/complaints', methods=['GET'])
def api_get_complaints():
    """
    API endpoint to get complaints for the dashboard.
    """
    complaints = get_complaints_for_dashboard()
    return jsonify(complaints)

@app.route('/', methods=['GET'])
def home():
    """Ana endpoint"""
    return render_template('index.html')

@app.route('/api/session', methods=['GET', 'OPTIONS'])
def get_active_session():
    """Yeni bir session başlat"""
    if request.method == 'OPTIONS':
        return jsonify({"status": "success"})
        
    new_session_id = user_memory.create_new_session()
    
    # Session ID'yi kontrol et
    if not new_session_id:
        return jsonify({
            "error": "Session oluşturulamadı"
        }), 500
        
    # Session bilgilerini döndür
    return jsonify({
        "status": "success",
        "session_id": new_session_id,
        "chat_endpoint": f"/api/chat/{new_session_id}",
        "message": "Session başarıyla oluşturuldu"
    })

@app.route('/api/chat/<session_id>', methods=['POST', 'OPTIONS'])
def chat_with_mem(session_id):
    """Chat endpoint'i"""
    if request.method == 'OPTIONS':
        return jsonify({"status": "success"})
        
    # Session ID kontrolü
    if not user_memory.is_valid_session(session_id):
        return jsonify({
            "error": "Geçersiz veya süresi dolmuş oturum",
            "message": "Lütfen yeni bir oturum başlatın"
        }), 401
    
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({
                "error": "Geçersiz istek",
                "message": "Request body'de 'message' alanı gerekli"
            }), 400
            
        user_message = data.get('message', '').strip().lower()
        
        # Kullanıcı mesajını geçmişe ekle
        user_memory.add_to_chat_history(session_id, user_message, "user")
        
        session_data = user_memory.get_session(session_id)
        context = session_data.get("context", {})
        
        # Fatura sorgulama kontrolleri
        if "fatura" in user_message or context.get("type", "").startswith("bill_query"):
            # Fatura sorgulama başlangıcı
            if not context.get("type"):
                user_memory.update_session(session_id, "context", {
                    "type": "bill_query_start",
                    "step": "waiting_identifier"
                })
                return jsonify({
                    "response": "Lütfen TC Kimlik Numaranızı (11 haneli) veya Abone Numaranızı (6 haneli) giriniz:",
                    "expecting": "identifier"
                })
            
            # TC veya Abone No girişi
            if context.get("step") == "waiting_identifier":
                identifier = ''.join(filter(str.isdigit, user_message))
                
                if len(identifier) == 11 and validate_tc(identifier):  # TC girildiyse
                    user_memory.update_session(session_id, "context", {
                        "type": "bill_query",
                        "step": "waiting_subscriber",
                        "tc_no": identifier
                    })
                    return jsonify({
                        "response": "Lütfen 6 haneli abone numaranızı giriniz:",
                        "expecting": "subscriber_no"
                    })
                elif len(identifier) == 6 and validate_subscriber_no(identifier):  # Abone No girildiyse
                    user_memory.update_session(session_id, "context", {
                        "type": "bill_query",
                        "step": "waiting_birth_date",
                        "subscriber_no": identifier
                    })
                    return jsonify({
                        "response": "Lütfen doğum tarihinizi YYYY-AA-GG formatında giriniz (Örnek: 1990-01-31):",
                        "expecting": "birth_date"
                    })
                else:
                    return jsonify({
                        "response": "Geçersiz giriş. Lütfen 11 haneli TC Kimlik Numarası veya 6 haneli Abone Numarası giriniz:",
                        "expecting": "identifier"
                    })
            
            # TC ile sorgulama - Abone No bekleniyor
            if context.get("step") == "waiting_subscriber":
                subscriber_no = ''.join(filter(str.isdigit, user_message))
                if validate_subscriber_no(subscriber_no):
                    user_memory.update_session(session_id, "context", {
                        "type": "bill_query",
                        "step": "waiting_birth_date",
                        "tc_no": context.get("tc_no"),
                        "subscriber_no": subscriber_no
                    })
                    return jsonify({
                        "response": "Lütfen doğum tarihinizi YYYY-AA-GG formatında giriniz (Örnek: 1990-01-31):",
                        "expecting": "birth_date"
                    })
            
            # Doğum tarihi bekleniyor
            if context.get("step") == "waiting_birth_date":
                if validate_date(user_message):
                    tc_no = context.get("tc_no")
                    subscriber_no = context.get("subscriber_no")
                    
                    # TC ile sorgulama
                    if tc_no:
                        bill_info = get_bill_info(tc_no, user_message, subscriber_no)
                    # Abone no ile sorgulama
                    else:
                        bill_info = get_bill_info(subscriber_no, user_message)
                    
                    if bill_info:
                        response = (
                            f"Fatura Bilgileriniz:\n"
                            f"Fatura Tutarı: {bill_info['bill_amount']} TL\n"
                            f"Kullanılan Su Miktarı: {bill_info['water_usage']} m³\n"
                            f"Fatura Tarihi: {bill_info['bill_date']}"
                        )
                    else:
                        response = "Bu bilgilerle eşleşen fatura bulunamadı."
                    
                    user_memory.update_session(session_id, "context", {})  # Context'i temizle
                    user_memory.add_to_chat_history(session_id, response, "bot")
                    return jsonify({"response": response})
                else:
                    return jsonify({
                        "response": "Geçersiz tarih formatı. Lütfen YYYY-AA-GG formatında giriniz (Örnek: 1990-01-31):",
                        "expecting": "birth_date"
                    })
        
        # Döviz kuru sorgusu
        elif any(word in user_message for word in ["döviz", "kur", "euro", "dolar", "sterlin", "usd", "eur", "gbp"]):
            exchange_info = get_exchange_rate()
            user_memory.update_session(session_id, "context", {})  # Context'i temizle
            user_memory.add_to_chat_history(session_id, exchange_info, "bot")
            return jsonify({"response": exchange_info})
        
        # Hava durumu sorgusu
        elif "hava" in user_message or "hava durumu" in user_message:
            city = extract_city_from_message(user_message)
            if city:
                weather_info = get_weather(city)
                user_memory.update_session(session_id, "context", {})  # Context'i temizle
                user_memory.add_to_chat_history(session_id, weather_info, "bot")
                return jsonify({"response": weather_info})
            else:
                user_memory.update_session(session_id, "context", {
                    "type": "weather_query",
                    "step": "waiting_city"
                })
                return jsonify({
                    "response": "Hangi şehir için hava durumu bilgisi istersiniz?",
                    "expecting": "city"
                })
        
        # Sadece şehir ismi yazıldıysa
        elif context.get("type") == "weather_query" and context.get("step") == "waiting_city":
            city = extract_city_from_message(user_message)
            if city:
                weather_info = get_weather(city)
                user_memory.update_session(session_id, "context", {})  # Context'i temizle
                user_memory.add_to_chat_history(session_id, weather_info, "bot")
                return jsonify({"response": weather_info})
            else:
                return jsonify({
                    "response": "Geçersiz şehir ismi. Lütfen Türkiye'deki bir şehir adı giriniz.",
                    "expecting": "city"
                })
        
        # Su kesintisi sorgusu
        elif "su kesintisi" in user_message or context.get("type") == "water_outage_query":
            if not context.get("type"):
                user_memory.update_session(session_id, "context", {
                    "type": "water_outage_query",
                    "step": "waiting_district"
                })
                return jsonify({
                    "response": "Hangi mahalle için su kesintisi bilgisi istersiniz?",
                    "expecting": "district"
                })
            
            # Mahalle bekleniyor
            if context.get("step") == "waiting_district":
                district = user_message.strip()
                outage_info = check_water_outage(district)
                user_memory.update_session(session_id, "context", {})  # Context'i temizle
                user_memory.add_to_chat_history(session_id, outage_info, "bot")
                return jsonify({"response": outage_info})
        
        # Sayaç bilgisi sorgulama
        elif "sayaç" in user_message and ("sorgula" in user_message or "bilgi" in user_message or "öğren" in user_message) or context.get("type") == "meter_query":
            if not context.get("type"):
                user_memory.update_session(session_id, "context", {
                    "type": "meter_query",
                    "step": "waiting_subscriber"
                })
                return jsonify({
                    "response": "Sayaç bilgilerinizi öğrenmek için lütfen 6 haneli abone numaranızı giriniz:",
                    "expecting": "subscriber_no"
                })
            
            current_step = context.get("step")
            
            # Abone no bekleniyor
            if current_step == "waiting_subscriber":
                subscriber_no = ''.join(filter(str.isdigit, user_message))
                if validate_subscriber_no(subscriber_no):
                    meter_info = get_meter_info(subscriber_no)
                    
                    if meter_info:
                        response = (
                            f"Sayaç Bilgileriniz:\n"
                            f"Sayaç ID: {meter_info['meter_id']}\n"
                            f"Son Okuma: {meter_info['last_reading']} m³\n"
                            f"Son Okuma Tarihi: {meter_info['last_reading_date']}\n"
                            f"Durum: {meter_info['status']}\n"
                            f"Kurulum Tarihi: {meter_info['installation_date']}\n"
                            f"Sayaç Tipi: {meter_info['meter_type']}\n"
                            f"Adres: {meter_info['address']}"
                        )
                    else:
                        response = "Bu abone numarasına ait sayaç bilgisi bulunamadı."
                    
                    user_memory.update_session(session_id, "context", {})  # Context'i temizle
                    user_memory.add_to_chat_history(session_id, response, "bot")
                    return jsonify({"response": response})
                else:
                    return jsonify({
                        "response": "Geçersiz abone numarası. Lütfen 6 haneli abone numaranızı giriniz:",
                        "expecting": "subscriber_no"
                    })
        
        # Su sayacı şikayet sistemi
        elif any(word in user_message for word in ["sayacım bozuk", "sayaç arıza", "sayaç bozuk", "sayaç şikayet"]) or "sayaç" in user_message and any(word in user_message for word in ["arıza", "bozuk", "çalışmıyor", "problem", "sorun", "şikayet"]) or context.get("type") == "meter_complaint":
            # Yeni şikayet başlat
            if not context.get("type"):
                user_memory.update_session(session_id, "context", {
                    "type": "meter_complaint",
                    "step": "waiting_complaint_type",
                    "complaint_data": {}
                })
                return jsonify({
                    "response": (
                        "Su sayacı şikayetinizi almak için lütfen şikayet türünü seçin:\n"
                        "1. Sayaç çalışmıyor\n"
                        "2. Sayaç arızalı/bozuk\n"
                        "3. Sayaç okuma hatası var\n"
                        "4. Sayaç değişimi istiyorum\n"
                        "5. Sayaçta fiziksel hasar var\n"
                        "6. Diğer\n"
                        "Lütfen seçiminizi 1-6 arasında bir rakam olarak belirtin:"
                    ),
                    "expecting": "complaint_type"
                })
            
            current_step = context.get("step")
            complaint_data = context.get("complaint_data", {})
            
            # Şikayet türü bekleniyor
            if current_step == "waiting_complaint_type":
                complaint_types = {
                    "1": "Sayaç çalışmıyor",
                    "2": "Sayaç arızalı/bozuk",
                    "3": "Sayaç okuma hatası",
                    "4": "Sayaç değişim talebi",
                    "5": "Sayaçta fiziksel hasar",
                    "6": "Diğer"
                }
                
                if user_message in complaint_types:
                    complaint_data["complaint_type"] = complaint_types[user_message]
                    context["complaint_data"] = complaint_data
                    context["step"] = "waiting_description"
                    user_memory.update_session(session_id, "context", context)
                    
                    return jsonify({
                        "response": "Lütfen sorunu kısaca açıklayınız:",
                        "expecting": "description"
                    })
                else:
                    return jsonify({
                        "response": "Geçersiz seçim. Lütfen 1-6 arasında bir rakam girin:",
                        "expecting": "complaint_type"
                    })
            
            # Açıklama bekleniyor
            elif current_step == "waiting_description":
                complaint_data["description"] = user_message
                context["complaint_data"] = complaint_data
                context["step"] = "waiting_subscriber"
                user_memory.update_session(session_id, "context", context)
                
                return jsonify({
                    "response": "Lütfen 6 haneli abone numaranızı giriniz:",
                    "expecting": "subscriber_no"
                })
            
            # Abone no bekleniyor
            elif current_step == "waiting_subscriber":
                subscriber_no = ''.join(filter(str.isdigit, user_message))
                if validate_subscriber_no(subscriber_no):
                    complaint_data["subscriber_no"] = subscriber_no
                    context["complaint_data"] = complaint_data
                    context["step"] = "waiting_birth_date"
                    user_memory.update_session(session_id, "context", context)
                    
                    return jsonify({
                        "response": "Lütfen doğum tarihinizi YYYY-AA-GG formatında giriniz (Örnek: 1990-01-31):",
                        "expecting": "birth_date"
                    })
            
            # Doğum tarihi bekleniyor
            elif current_step == "waiting_birth_date":
                if validate_date(user_message):
                    complaint_data["birth_date"] = user_message
                    context["complaint_data"] = complaint_data
                    context["step"] = "waiting_address"
                    user_memory.update_session(session_id, "context", context)
                    
                    return jsonify({
                        "response": "Lütfen açık adresinizi giriniz:",
                        "expecting": "address"
                    })
                else:
                    return jsonify({
                        "response": "Geçersiz tarih formatı. Lütfen YYYY-AA-GG formatında giriniz (Örnek: 1990-01-31):",
                        "expecting": "birth_date"
                    })
            
            # Adres bekleniyor
            elif current_step == "waiting_address":
                complaint_data["address"] = user_message
                
                # Tüm gerekli bilgilerin var olduğunu kontrol et
                required_fields = ["complaint_type", "description", "subscriber_no", "birth_date", "address"]
                if all(key in complaint_data for key in required_fields):
                    # Şikayeti kaydet
                    complaint_id = save_complaint(complaint_data)
                    
                    if complaint_id:
                        response = (
                            f"Şikayetiniz başarıyla kaydedildi.\n"
                            f"Şikayet Numaranız: #{complaint_id}\n"
                            f"Şikayet Türü: {complaint_data['complaint_type']}\n"
                            f"İlgili birimlerimiz en kısa sürede sizinle iletişime geçecektir.\n"
                            f"Bizi bilgilendirdiğiniz için teşekkür ederiz."
                        )
                    else:
                        response = "Şikayetiniz kaydedilirken bir hata oluştu. Lütfen daha sonra tekrar deneyiniz."
                    
                    # Context'i temizle
                    user_memory.update_session(session_id, "context", {})
                    user_memory.add_to_chat_history(session_id, response, "bot")
                    return jsonify({"response": response})
                else:
                    # Eksik bilgi varsa baştan başlat
                    user_memory.update_session(session_id, "context", {
                        "type": "meter_complaint",
                        "step": "waiting_complaint_type",
                        "complaint_data": {}
                    })
                    return jsonify({
                        "response": "Bazı bilgiler eksik. Lütfen tekrar başlayalım.\nLütfen şikayet türünü seçin (1-6):",
                        "expecting": "complaint_type"
                    })
        
        # Saat sorgusu
        elif any(word in user_message for word in ["saat kaç", "saat", "saat kaç?"]):
            current_time = get_current_time()
            return jsonify({
                "response": f"Şu anki saat: {current_time}"
            })
        
        # Diğer durumlar için chatbot'a sor
        else:
            # Desteklenen özellikler listesi
            supported_features = [
                "fatura sorgulama",
                "sayaç bilgisi sorgulama",
                "sayaç arıza bildirimi",
                "su kesintisi sorgulama",
                "döviz kuru bilgisi",
                "hava durumu",
                "saat bilgisi"
            ]

            # Chatbot'a sor
            response, intent, confidence = chatbot.predict(user_message.upper())
            if response and confidence >= chatbot.confidence_threshold:
                user_memory.add_to_chat_history(session_id, response, "bot")
                return jsonify({
                    "response": response,
                    "intent": intent,
                    "confidence": confidence
                })
            else:
                # Anlaşılamayan soru için yardımcı yanıt
                help_response = (
                    "Sorduğunuz soruyu anlayamadım. Soruyu başka şekilde sormak ister misiniz?\n\n"
                    "Size şu konularda yardımcı olabilirim:\n"
                    "🔹 " + "\n🔹 ".join(supported_features) + "\n\n"
                    "Lütfen bu konulardan biriyle ilgili soru sorun."
                )
                return jsonify({
                    "response": help_response,
                    "confidence": confidence
                })
        
    except Exception as e:
        logging.error(f"API hatası: {e}")
        return jsonify({
            "error": "Bir hata oluştu",
            "details": str(e)
        }), 500

# Konuşma geçmişini getirmek için yeni endpoint
@app.route('/api/chat/<session_id>/history', methods=['GET'])
def get_chat_history(session_id):
    """Konuşma geçmişini getir"""
    if not user_memory.is_valid_session(session_id):
        return jsonify({
            "error": "Geçersiz veya süresi dolmuş oturum"
        }), 401
    
    chat_history = user_memory.get_user_chat_history(session_id)
    return jsonify({
        "chat_history": chat_history
    })

def extract_city_from_message(message):
    """Mesajdan şehir ismini çıkar"""
    message = message.lower()
    for city in TURKEY_CITIES:
        if city in message:
            return city
    return None

def get_meter_info(subscriber_no):
    """Sayaç bilgilerini getir"""
    try:
        meter_info = METER_DATABASE.get(subscriber_no)
        if meter_info:
            return meter_info
        return None
    except Exception as e:
        logging.error(f"Sayaç bilgisi getirme hatası: {e}")
        return None

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
            inputs = self.tokenizer(
                text,
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

# Konuşma geçmişini getirmek için yeni endpoint'ler
@app.route('/api/chat/history/user/<user_id>', methods=['GET'])
def get_user_history(user_id):
    """Kullanıcının tüm konuşma geçmişini getir"""
    date = request.args.get('date')  # YYYYMMDD formatında tarih parametresi
    
    chat_history = user_memory.get_user_chat_history(user_id, date)
    return jsonify({
        "user_id": user_id,
        "date": date,
        "chat_history": chat_history
    })

@app.route('/api/complaints/<int:complaint_id>/resolve', methods=['POST'])
def resolve_complaint(complaint_id):
    """
    API endpoint to mark a complaint as resolved.
    """
    complaints = load_complaints_from_json('complaints.json')
    for complaint in complaints:
        if complaint['id'] == complaint_id:
            complaint['status'] = 'Çözüldü'
            break
    else:
        return jsonify({"error": "Complaint not found"}), 404

    # Save the updated complaints back to the JSON file
    save_complaints_to_json(complaints)
    return jsonify({"message": "Complaint resolved successfully"})

def save_complaints_to_json(complaints, file_path='solved_complaint.json'):
    """
    Save complaints to a JSON file.
    
    :param complaints: List of complaints.
    :param file_path: Path to the JSON file.
    """
    import json
    
    with open(file_path, 'w') as file:
        json.dump({"complaints": complaints}, file, indent=4)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)