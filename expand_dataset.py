import json
import requests
from bs4 import BeautifulSoup
import re
import random

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def scrape_forum_data(url, num_pages=5):
    forum_data = []
    for i in range(1, num_pages + 1):
        page_url = f"{url}?page={i}"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        questions = soup.find_all('div', class_='question')
        for question in questions:
            title = question.find('h3').text.strip()
            content = question.find('div', class_='content').text.strip()
            forum_data.append({'question': title, 'content': content})
    return forum_data

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()

def generate_synthetic_data(examples, n=100):
    synthetic_data = []
    for _ in range(n):
        example = random.choice(examples)
        words = example.split()
        # Randomly remove, duplicate, or shuffle words
        if len(words) > 3:
            if random.choice([True, False]):
                words.pop(random.randint(0, len(words) - 1))
            else:
                words.insert(random.randint(0, len(words)), random.choice(words))
        random.shuffle(words)
        synthetic_data.append(' '.join(words))
    return synthetic_data

def main():
    input_file = 'chatbot_data.json'
    output_file = 'expanded_chatbot_data.json'
    forum_url = 'https://www.koski.gov.tr/koski/ss-sorular'  # Örnek bir Türkçe forum URL'si

    # Mevcut veriyi yükle
    data = load_data(input_file)

    # Forum verisi topla
    print("Forum verileri toplanıyor...")
    forum_data = scrape_forum_data(forum_url)

    # Yeni veriyi mevcut veri yapısına ekle
    for item in forum_data:
        intent_name = clean_text(item['question'])[:50]  # İlk 50 karakter
        new_intent = {
            "intent": intent_name,
            "examples": [clean_text(item['question']), clean_text(item['content'])],
            "responses": ["Bu konu hakkında daha fazla bilgi verebilirim. Ne öğrenmek istersiniz?"]
        }
        data['intents'].append(new_intent)

    # Her niyet için sentetik veri oluştur
    print("Sentetik veriler oluşturuluyor...")
    for intent in data['intents']:
        synthetic_examples = generate_synthetic_data(intent['examples'])
        intent['examples'].extend(synthetic_examples)

    # Genişletilmiş veriyi kaydet
    save_data(data, output_file)

    print(f"Orijinal örnek sayısı: {sum(len(intent['examples']) for intent in load_data(input_file)['intents'])}")
    print(f"Genişletilmiş örnek sayısı: {sum(len(intent['examples']) for intent in data['intents'])}")

if __name__ == "__main__":
    main()