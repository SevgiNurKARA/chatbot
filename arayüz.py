import tkinter as tk
from tkinter import scrolledtext
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import json
import requests
from datetime import datetime

class ChatbotInterface:
    def __init__(self, model, tokenizer, label_mapping):
        self.model = model
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        with open('chatbot_data.json', 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.kvkk_accepted = False
        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Yapay Zeka Chatbot")

        # KVKK Metni
        self.kvkk_text = tk.Label(
            self.root,
            text="KVKK Metni: Sohbeti başlatmadan önce onay vermeniz gerekmektedir.",
            wraplength=400,
            font=("Arial", 10)
        )
        self.kvkk_text.pack(pady=10)

        # KVKK Onay Butonları
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.yes_button = tk.Button(
            self.button_frame, 
            text="Evet", 
            command=self.accept_kvkk, 
            font=("Arial", 12), 
            bg="green", 
            fg="white"
        )
        self.yes_button.pack(side=tk.LEFT, padx=5)

        self.no_button = tk.Button(
            self.button_frame, 
            text="Hayır", 
            command=self.reject_kvkk, 
            font=("Arial", 12), 
            bg="red", 
            fg="white"
        )
        self.no_button.pack(side=tk.LEFT, padx=5)

        # Sohbet penceresi
        self.chat_window = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            width=50, 
            height=15, 
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.chat_window.pack(pady=10)

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
