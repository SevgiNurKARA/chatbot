import tkinter as tk
from tkinter import scrolledtext
import tkinter as tk
from tkinter import messagebox
from input_extraction.input_extraction import get_output
import requests
from datetime import datetime

 
def get_weather(city):
    api_key = "7549d0b1ceff6a105a44074f3df577e9"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=tr"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        weather_data = response.json()
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        return f"Sıcaklık: {temp}°C\nDurum: {description}"
    except:
        return "Hava durumu bilgisi alınamadı."
    
def submit():
    subscriber_input = entry_subscriber.get()
    city = entry_city.get()
    
    # Abone bilgisi sorgulama
    subscriber_result = get_output(subscriber_input)
    
    # Hava durumu sorgulama
    weather_result = get_weather(city)
    
    # Sonuçları göster
    messagebox.showinfo("Sonuç", f"Abone Bilgisi:\n{subscriber_result}\n\nHava Durumu:\n{weather_result}")

# Ana pencere oluşturma
root = tk.Tk()
root.title("Bilgi Sorgulama Sistemi")
root.geometry("400x200")

# Abone numarası girişi
tk.Label(root, text="Abone Numaranız:").pack(pady=5)
entry_subscriber = tk.Entry(root)
entry_subscriber.pack(pady=5)

# Şehir girişi
tk.Label(root, text="Şehir:").pack(pady=5)
entry_city = tk.Entry(root)
entry_city.pack(pady=5)

# Sorgula butonu
tk.Button(root, text="Sorgula", command=submit).pack(pady=20)

root.mainloop()

<<<<<<< HEAD
=======
        # Kullanıcı girişi
        self.user_entry = tk.Entry(
            self.root, 
            font=("Arial", 12), 
            width=40,
            state=tk.DISABLED
        )
        self.user_entry.pack(pady=5)

        # Gönder düğmesi
        self.send_button = tk.Button(
            self.root, 
            text="Gönder", 
            command=self.send_message, 
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.send_button.pack()

    def accept_kvkk(self):
        self.kvkk_accepted = True
        # KVKK metnini kaldır
        self.kvkk_text.pack_forget()
        # Butonları kaldır
        self.yes_button.pack_forget()
        self.no_button.pack_forget()
        self.button_frame.pack_forget()
        # Sohbet penceresini ve girişleri aktif et
        self.chat_window.config(state=tk.NORMAL)
        self.user_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, "Chatbot: KVKK onayınız için teşekkür ederiz! Sorularınızı sorabilirsiniz.\n\n")

    def reject_kvkk(self):
        self.kvkk_accepted = False
        # KVKK metnini kaldır
        self.kvkk_text.pack_forget()
        # Butonları kaldır
        self.yes_button.pack_forget()
        self.no_button.pack_forget()
        self.button_frame.pack_forget()
        # Sohbet penceresini devre dışı bırak
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, "Chatbot: Üzgünüm, onay olmadan işlem yapamıyorum.\n")
        self.chat_window.config(state=tk.DISABLED)

    def send_message(self):
        if not self.kvkk_accepted:
            return

        user_input = self.user_entry.get()
        if user_input:
            self.chat_window.insert(tk.END, f"Kullanıcı: {user_input}\n")
            response = self.get_model_response(user_input)
            self.chat_window.insert(tk.END, f"Chatbot: {response}\n\n")
            self.user_entry.delete(0, tk.END)

    def get_model_response(self, user_input):
        try:
            # Saat sorusu
            if "saat" in user_input.lower():
                return f"Şu an saat: {datetime.now().strftime('%H:%M:%S')}"

            # Hava durumu sorusu
            if "hava" in user_input.lower():
                return self.get_weather_response()

            # NLP modeliyle yanıt
            inputs = self.tokenizer(user_input, truncation=True, padding=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            predicted_label = self.reverse_label_mapping[predicted_class]
            for item in self.dataset['intents']:
                if item['intent'] == predicted_label:
                    responses = item.get('responses', [])
                    if responses:
                        import random
                        return random.choice(responses)
                    return "Bu intent için bir yanıt tanımlanmamış."
            return "Üzgünüm, bu konuda bir yanıt bulamadım."
        except Exception as e:
            return f"Yanıt alınamadı: {str(e)}"

    def get_weather_response(self):
        city = "Konya"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=tr"

        try:
            response = requests.get(url)
            weather_data = response.json()
            if weather_data.get("main"):
                temp = weather_data["main"]["temp"]
                weather_desc = weather_data["weather"][0]["description"]
                return f"{city} için hava durumu: {weather_desc}, sıcaklık: {temp}°C"
            else:
                return "Hava durumu alınamadı."
        except Exception as e:
            return f"Hava durumu sorgulanırken bir hata oluştu: {str(e)}"

    def run(self):
        self.root.mainloop()

def load_your_model():
    model = DistilBertForSequenceClassification.from_pretrained('./saved_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained('./saved_model')
    with open('./saved_model/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    return model, tokenizer, label_mapping

def main():
    model, tokenizer, label_mapping = load_your_model()
    chatbot_ui = ChatbotInterface(model, tokenizer, label_mapping)
    chatbot_ui.run()

if __name__ == "__main__":
    main()
>>>>>>> 3a4d28dfdd90409d669598fd3057b2c9d9ce88f3
