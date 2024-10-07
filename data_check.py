def read_json_file(file_path):
    try:
        # Open the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except UnicodeDecodeError:
        # If UTF-8 fails, try with utf-8-sig (handles BOM)
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return data
        except UnicodeDecodeError as e:
            print(f"Error reading the file: {e}")
            print("Try checking the file encoding or converting it to UTF-8")
            return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Usage
import json
file_path = "chatbot_data.json"
data = read_json_file(file_path)

if data is not None:
    print("Successfully loaded JSON data")
    # Process your data here
import json

# JSON dosyasını aç ve yükle
with open('chatbot_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Örnek olarak bir niyeti al ve göster
for intent in data['intents']:
    print(f"Niyet: {intent['intent']}")
    print(f"Sorular: {intent['examples']}")
    print(f"Yanıtlar: {intent['responses']}")
