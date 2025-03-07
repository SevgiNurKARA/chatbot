# DAMLA Chatbot

DAMLA, Türkçe dil desteği ile geliştirilmiş bir akıllı sohbet robotu (chatbot) uygulamasıdır. Bu uygulama, kullanıcı sorguları için doğal dil işleme (NLP) teknolojileri kullanarak uygun yanıtlar verecek şekilde tasarlanmıştır.

## Proje Yapısı

Proje üç ana bileşenden oluşmaktadır:

1. **Eğitim Modülü**: BERT tabanlı bir sınıflandırma modeli kullanarak intent (niyet) tanıma işlemlerini gerçekleştirir.
2. **Web Arayüzü**: Kullanıcıların chatbot ile etkileşime geçebileceği interaktif bir arayüz sunar.
3. **Yönetim Paneli**: Chatbot verilerini, model eğitimini ve sistem loglarını yönetmek için React tabanlı bir panel.

## Teknik Detaylar

### Model Eğitimi

- `ChatbotTrainer` sınıfı, chatbot için NLP modelini eğitir
- Türkçe diline optimize edilmiş `dbmdz/bert-base-turkish-uncased` modeli kullanılmaktadır
- Veri genişletme teknikleri ile daha zengin bir eğitim seti oluşturulur
- Veri yükleme, hazırlama, eğitim ve kaydetme işlemleri tek bir akışta gerçekleştirilir

### Model Özellikleri

- Intent sınıflandırma için PyTorch ve Hugging Face Transformers
- Veri işleme için Scikit-learn ve Datasets kütüphaneleri
- GPU desteği (CUDA mevcutsa otomatik kullanılır)
- Model performansını değerlendirmek için doğruluk ve F1 metrikleri

### Web Arayüzü

- Duyarlı tasarım (responsive design) ile mobil uyumlu arayüz
- Animasyonlu robot karakteri
- Kullanıcı dostu sohbet arayüzü
- KVKK (Kişisel Verilerin Korunması Kanunu) onay mekanizması
- Oturum yönetimi ve API entegrasyonu

### Yönetim Paneli

- React ve React Router tabanlı tek sayfalı uygulama (SPA)
- Ana bileşenler:
  - **Home**: Genel durum bilgisi ve özet istatistikler
  - **EnterData**: Kullanıcı şikayet ve önerilerinin girişi ve model fine-tuning ayarları
  - **LogViewer**: Sistem loglarının görüntülenmesi ve analizi
- Genişletilebilir yapı (yorumlanmış kodda planlanan ek sayfalar bulunmaktadır)

## Kurulum

### Gereksinimler

```
# Backend
torch
transformers
scikit-learn
datasets
numpy
flask

# Frontend
react
react-router-dom
node.js (v14+)
```

### Model Eğitimi

1. `chatbot_data.json` dosyasını projenin kök dizinine yerleştirin
2. Eğitim scriptini çalıştırın:

```bash
python chatbot_trainer.py
```

3. Eğitim tamamlandığında model `saved_model` dizinine kaydedilecektir

### Web Arayüzünün Çalıştırılması

Web arayüzü için bir Flask uygulaması kullanılmaktadır:

```bash
flask run
```

### Yönetim Panelinin Çalıştırılması

React uygulamasını başlatmak için:

```bash
# Proje klasörüne girin
cd yonetim-paneli

# Bağımlılıkları yükleyin
npm install

# Geliştirme sunucusunu başlatın
npm start
```

## API Kullanımı

Web arayüzü aşağıdaki API endpointleri ile iletişim kurar:

- `/api/session`: Yeni bir sohbet oturumu başlatır
- `/api/chat/{session_id}`: Mesaj gönderme ve yanıt alma

Yönetim paneli aşağıdaki endpointleri kullanır:

- `/api/logs`: Sistem loglarını alır
- `/api/feedback`: Kullanıcı geri bildirimlerini alır ve kaydeder
- `/api/training`: Model yeniden eğitimini başlatır

## Kullanıcı Deneyimi

1. Robot karakterine tıklayarak sohbeti başlatın
2. KVKK aydınlatma metnini kabul edin
3. Metin kutusuna mesajınızı yazın ve gönderin
4. Chatbot yanıtını bekleyin

## Yönetim Paneli Kullanımı

1. `/data-girisi` sayfası ile yeni veriler ekleyerek modelin eğitim veri setini zenginleştirin
2. `/logs` sayfasından sistem loglarını inceleyerek hatalar veya optimizasyon fırsatları tespit edin
3. Gelecekte eklenecek dashboard, kullanıcı yönetimi vb. sayfalar ile ek özelliklere erişim sağlanacaktır

## Özelleştirme

- Robot görselini değiştirmek için yeni GIF dosyasını `static` klasörüne ekleyin
- CSS stilleri HTML içindeki `<style>` etiketleri arasında düzenlenebilir
- Model yanıtlarını değiştirmek için `chatbot_data.json` dosyasını güncelleyin ve modeli yeniden eğitin
- Yönetim panelinde yeni sayfalar eklemek için `App.js` dosyasındaki yorumlanmış Route bileşenlerini aktif hale getirin

Projenin çalışma şekliyle ilgili qr kodu tarayınız!

  ![frame](https://github.com/user-attachments/assets/cb1efb57-e0b1-4723-9a91-245232cad765)

