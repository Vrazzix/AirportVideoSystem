import requests

url = "http://localhost:5678/webhook-test/detect-result"

# Текстовые данные теперь передаем отдельно
data = {
    "image_name": "frame_042.jpg",
    "detected_parts": "door, hatch, person", # Для form-data список лучше передать строкой
    "confidence_score": "0.92"
}

# Открываем файл в режиме бинарного чтения ('rb')
# 'my_photo' - это ключ, по которому n8n поймет, где искать картинку
files = {
    'my_photo': open('C:\\Users\\shche\\Desktop\\Application_for_models\\auto_labeled\\visualizations\\high_confidence_chock_3bfcffedfdd0.jpg', 'rb') 
}

response = requests.post(url, data=data, files=files)
print("Отправлено!", response.status_code)