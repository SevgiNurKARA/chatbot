import openai

openai.api_key = 'sk-5T4uAnXjWmbpqfhxRfP_iFBCnF_Us_CfOlTCcRfKyeT3BlbkFJN1UUlJb0U61loiCz8C7b2fsemklOXPgjyiaq7auuQA'

def get_response(prompt):
    response = openai.Completion.create(
        engine="gpt-4",  # veya "gpt-3.5-turbo"
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Örnek bir sorgu
sorgu = "Faturamda hata var, ne yapmalıyım?"
print(get_response(sorgu))
