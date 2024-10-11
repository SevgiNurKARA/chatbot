import json
from sklearn.model_selection import train_test_split

# Veri setini yükleme
with open('chatbot_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train_data = []
test_data = []

# Her intent için examples ve responses'ları ayır
for intent_data in data['intents']:
    examples = intent_data['examples']
    responses = intent_data['responses']
    
    # Veriyi train ve test olarak ayır (%80 train, %20 test)
    train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
    
    # Eğitim verisine ekle
    train_data.append({
        "intent": intent_data['intent'],
        "examples": train_examples,
        "responses": responses  # Tüm yanıtları hem eğitim hem testte kullanabiliriz
    })
    
    # Test verisine ekle
    test_data.append({
        "intent": intent_data['intent'],
        "examples": test_examples,
        "responses": responses  # Aynı şekilde yanıtlar testte de bulunabilir
    })

# Sonuçları iki ayrı dosyaya kaydet
with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("Veri başarıyla train ve test olarak ayrıldı.")
