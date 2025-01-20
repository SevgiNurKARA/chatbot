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

