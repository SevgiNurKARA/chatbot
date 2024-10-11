import json
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# Eğitim ve test verilerini yükle
with open('train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Örnekleri ve intent'leri ayırma
def prepare_data(data):
    examples = []
    labels = []
    for item in data:
        for example in item['examples']:
            examples.append(example)
            labels.append(item['intent'])
    return examples, labels

# Eğitim ve test setlerini hazırla
train_texts, train_labels = prepare_data(train_data)
test_texts, test_labels = prepare_data(test_data)

# LabelEncoder kullanarak etiketleri (intentleri) sayısal değerlere dönüştür
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Sayısallaştırılmış etiketlerle Dataset formatına dönüştür
train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels_encoded})
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels_encoded})

# Dönüşüm tamamlandı
print(train_dataset)
print(test_dataset)
