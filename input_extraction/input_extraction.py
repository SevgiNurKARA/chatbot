import re
import requests

# 1. Adım: Abone numarasını çıkarma
def extract_subscriber_number(input_text):
    match = re.search(r'\d+', input_text)
    if match:
        return int(match.group(0))  # Numarayı tam sayı olarak döndür
    return None

# 2. Adım: Fake API'ye istek gönderme (GET isteğiyle)
def get_fake_api_data(subscriber_number):
    url = f"https://jsonplaceholder.typicode.com/posts/{subscriber_number}"
    try:
        # API'den abone numarasına karşılık gelen veri almak için
        response = requests.get(url)
        response.raise_for_status()  # HTTP hata kodları için exception fırlatır
        data = response.json()  # JSON verisini alıyoruz
        
        # Veriyi kontrol etme
        if 'body' in data:
            body = data.get("body", "Body bulunamadı")
            return body
        else:
            return "Veri bulunamadı, geçerli bir abone numarası girin."
    except requests.exceptions.RequestException as e:
        return f"API'ye bağlanılamadı: {e}"

# 3. Adım: API cevabını işleme
def get_output(input_text):
    subscriber_number = extract_subscriber_number(input_text)
    if subscriber_number:
        body = get_fake_api_data(subscriber_number)
        return body
    else:
        return "Abone numarası bulunamadı."

# Örnek kullanım
input_text = "Abone numaram 2"
print(get_output(input_text))
