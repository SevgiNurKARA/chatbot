import re
import requests

def extract_subscriber_number(input_text):
    match = re.search(r'\d+', input_text)
    if match:
        return int(match.group(0)) 
    return None
def get_fake_api_data(subscriber_number):
    url = f"https://jsonplaceholder.typicode.com/posts/{subscriber_number}"
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()  
        
        
        if 'body' in data:
            body = data.get("body", "Body bulunamadı")
            return body
        else:
            return "Veri bulunamadı, geçerli bir abone numarası girin."
    except requests.exceptions.RequestException as e:
        return f"API'ye bağlanılamadı: {e}"

def get_output(input_text):
    subscriber_number = extract_subscriber_number(input_text)
    if subscriber_number:
        body = get_fake_api_data(subscriber_number)
        return body
    else:
        return "Abone numarası bulunamadı."

# Örnek kullanım
input_text = "Abone numaram 42"
print(get_output(input_text))
