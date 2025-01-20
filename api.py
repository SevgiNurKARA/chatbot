from flask import Flask, request, jsonify
import requests
from datetime import datetime

app = Flask(__name__)

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
def get_weather(city):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=tr"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        weather_data = response.json()
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        return f"{city} için hava durumu:\nSıcaklık: {temp}°C\nDurum: {description}"
    except:
        return "Hava durumu bilgisi alınamadı."

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
    data = request.json
    user_message = data.get('message', '').lower()
    
    # Hava durumu kontrolü
    if "hava" in user_message:
        found_city = None
        # Eğer mesajda şehir adı varsa direkt hava durumunu göster
        for city in TURKEY_CITIES:  # Örnek şehir listesi
            if city in user_message.lower():
                found_city = city
                break
        
        if found_city:
            weather_info = get_weather(found_city)
            return jsonify({"response": weather_info})
        
        # Şehir bulunamadıysa şehir sor
        return jsonify({
            "response": "Hangi şehir için hava durumu bilgisi istersiniz?",
            "expecting": "city"
        })
    
    # Şehir yanıtı geldiğinde
    if data.get('expecting') == 'city':
        city = user_message.strip().lower()  # Boşlukları temizle ve küçük harfe çevir
        
        # Şehir adını kontrol et
        if city in TURKEY_CITIES:
            weather_info = get_weather(city)
            return jsonify({"response": weather_info})
        else:
            return jsonify({"response": "Geçerli bir şehir adı giriniz."})
    
    # Saat sorgusu
    if "saat" in user_message:
        current_time = get_current_time()
        return jsonify({"response": f"Şu anki saat: {current_time}"})
    
    # Abone numarası kontrolü
    if "abone" in user_message or "endeks" in user_message:
        # Mesajdan sayıları çıkar
        subscriber_id = ''.join(filter(str.isdigit, user_message))
        if subscriber_id:
            subscriber_info = get_subscriber_info(subscriber_id)
            response = f"Abone Bilgileri:\nİsim: {subscriber_info.get('name', 'Bulunamadı')}\nDurum: {subscriber_info.get('status', 'Bulunamadı')}\nPlan: {subscriber_info.get('plan', 'Bulunamadı')}"
            return jsonify({"response": response})
    
    # Eğer hiçbir özel durum yoksa
    return jsonify({"response": "Üzgünüm, bu mesajı anlayamadım. Hava durumu, saat veya abone bilgisi sorabilirsiniz."})

if __name__ == '__main__':
    app.run(debug=True)