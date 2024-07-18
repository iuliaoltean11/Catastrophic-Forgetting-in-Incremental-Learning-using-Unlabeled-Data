import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Definirea transformărilor pentru setul de date de testare
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Încărcarea setului de date de testare
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inițializarea modelului
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # Ajustăm la numărul de clase curente (0-4 pentru sarcina a doua remapate)

model = model.to(device)

# Încărcarea modelului antrenat
model.load_state_dict(torch.load('./ct_model_task2.pth'))
model.eval()  # Setăm modelul în modul de evaluare

# Funcție pentru evaluarea acurateței
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Evaluarea modelului
accuracy = evaluate_model(model, test_loader)
print(f'Acuratețea modelului pe setul de date de testare: {accuracy:.2f}%')

# Funcție pentru realizarea predicțiilor pe imagini noi
def predict(model, image_path):
    from PIL import Image
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Exemplu de utilizare a funcției de predicție
image_path = ''  # Schimbați cu calea către o imagine de test
predicted_class = predict(model, image_path)
print(f'Clasa prezisă pentru imaginea {image_path}: {predicted_class}')
