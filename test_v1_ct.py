import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


"""
 Antrenează modelul pe datele de antrenament pentru un număr specificat de epoci. 
 La fiecare epocă, calculează pierderea (loss), realizează backpropagation și actualizează parametrii modelului.
"""
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

"""
Funcție pentru ajustarea fină a modelului
Ajustează fin modelul antrenat pentru un număr specificat de epoci, folosind aceleași principii ca train_model.
"""

def fine_tune_model(model, train_loader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Fine-tuning Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

"""
     Funcție pentru distilarea cunoștințelor
    Realizează distilarea cunoștințelor de la modelul profesor (teacher) la modelul student. 
    Utilizează pierderea KL-Divergence, pentru a compara predicțiile modelului student
     cu cele ale modelului profesor și pierderea Cross Entropy pentru datele etichetate.
"""
def distill_knowledge(student, teacher, unlabeled_loader, labeled_loader, temperature=3, alpha=0.7, num_epochs=50):
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.0001)
    student.train()
    teacher.eval()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for (inputs_unlabeled, _), (inputs_labeled, labels_labeled) in zip(unlabeled_loader, labeled_loader):
            inputs_unlabeled, inputs_labeled, labels_labeled = inputs_unlabeled.to(device), inputs_labeled.to(
                device), labels_labeled.to(device)
            optimizer.zero_grad()

            # predicțiile soft ale modelului profesor (obținute pe date nelabelate) pot oferi informații valoroase pentru antrenarea modelului student.
            student_outputs_unlabeled = student(inputs_unlabeled)
            student_outputs_labeled = student(inputs_labeled)

            # dezactivează calculul gradientului, ceea ce reduce consumul de memorie și timpul de calcul
            with torch.no_grad():
                teacher_outputs = teacher(inputs_unlabeled) #verifica predictiile modelului profesor

            # Împărțirea output-urilor la o valoare temperature are rolul de a "înmui" (soften) distribuțiile de probabilitate. Temperaturile > fac distribuțiile mai uniforme.
            soft_target_loss = kl_div_loss(F.log_softmax(student_outputs_unlabeled / temperature, dim=1),
                                           F.softmax(teacher_outputs / temperature, dim=1)) * (
                                           temperature * temperature) #transf. scoruri brute in probab.
            hard_target_loss = ce_loss(student_outputs_labeled, labels_labeled) # Cross Entropy
            loss = soft_target_loss * alpha + hard_target_loss * (1.0 - alpha) # Combinarea celor două pierderi

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f'Distillation Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(unlabeled_loader) + len(labeled_loader)):.4f}')


# Filtrarea claselor
def filter_classes(dataset, classes):
    targets = torch.tensor(dataset.targets)
    print(f'Total samples before filtering: {len(targets)}')
    indices = [i for i, target in enumerate(targets) if target in classes]
    filtered_targets = targets[indices]
    print(f'Selected {len(indices)} samples for classes {classes}')
    dataset.targets = filtered_targets.tolist()
    dataset.data = dataset.data[indices]
    print(f'Filtered dataset size: {len(dataset.data)} samples for classes {classes}')
    return dataset


# Re-map classes 5-9 to 0-4
def remap_targets(dataset, class_mapping):
    dataset.targets = [class_mapping[target] for target in dataset.targets]
    return dataset


# transformări sunt utilizate pentru a pregăti datele CIFAR-10 pentru a fi utilizate cu un model pre-antrenat, cum ar fi ResNet18,
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Încărcarea datasetului
original_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Copierea datasetului original pentru a nu modifica datasetul inițial
train_dataset_0_4 = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_dataset_5_9 = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# Initializează modelele și optimizatorii
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clase pentru prima și a doua fază
first_task_classes = [0, 1, 2, 3, 4]
second_task_classes = [5, 6, 7, 8, 9]

# Verificare și afișare clase în dataset
all_classes = set(original_dataset.targets)
print("Classes in the dataset:", all_classes)

# Filtrarea datasetului pentru a păstra doar clasele dorite
first_task_dataset = filter_classes(train_dataset_0_4, first_task_classes)
second_task_dataset = filter_classes(train_dataset_5_9, second_task_classes)

# Re-map classes 5-9 to 0-4
class_mapping = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
second_task_dataset = remap_targets(second_task_dataset, class_mapping)

# Verificăm dacă dataseturile nu sunt goale
assert len(first_task_dataset.data) > 0, "First task dataset is empty. Check the classes and dataset."
assert len(second_task_dataset.data) > 0, "Second task dataset is empty. Check the classes and dataset."

# Pregătim seturile de date pentru antrenament
first_task_loader = DataLoader(first_task_dataset, batch_size=64, shuffle=True)
second_task_loader = DataLoader(second_task_dataset, batch_size=64, shuffle=True)

# Utilizarea unui subset mai mic din datasetul original pentru date nelabelate
subset_indices_unlabeled = np.random.choice(len(original_dataset.data), size=500, replace=False)
unlabeled_data = original_dataset.data[subset_indices_unlabeled]
unlabeled_targets = torch.tensor(original_dataset.targets)[subset_indices_unlabeled].tolist()

unlabeled_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
unlabeled_dataset.data = unlabeled_data
unlabeled_dataset.targets = unlabeled_targets
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

# Reducerea datelor din PT la 500 de mostre
subset_indices_pt = np.random.choice(len(first_task_dataset.data), size=500, replace=False)
pt_data = first_task_dataset.data[subset_indices_pt]
pt_targets = torch.tensor(first_task_dataset.targets)[subset_indices_pt].tolist()

pt_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
pt_dataset.data = pt_data
pt_dataset.targets = pt_targets
pt_loader = DataLoader(pt_dataset, batch_size=64, shuffle=True)

# Utilizarea unui model pre-antrenat (ResNet18)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(second_task_classes))

model = model.to(device)

# Congelarea straturilor inițiale ale modelului
# Congelarea straturilor inițiale ale modelului pre-antrenat, păstrând astfel greutățile învățate pe ImageNet.
# Aceasta permite antrenarea doar a straturilor finale, care sunt specifice sarcinii noastre.
for param in model.parameters():
    param.requires_grad = False

# Dezghețarea doar a straturilor finale
#Permite ajustarea fină doar a straturilor finale ale modelului, care sunt specifice sarcinii noastre. fine-tuning
for param in model.fc.parameters():
    param.requires_grad = True

# Pierderi și optimizatori
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Antrenare pe sarcina curentă (clasele 5-9)
print("Începem antrenarea pe sarcina curentă (clasele 5-9)")
train_model(model, second_task_loader, criterion, optimizer, num_epochs=50)

# Dezghețarea tuturor straturilor și ajustarea fină a întregului model
for param in model.parameters():
    param.requires_grad = True

#  va actualiza toți parametrii modelului.
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Ajustare fină a modelului pe toate datele
print("Începem ajustarea fină a modelului")
fine_tune_model(model, second_task_loader, criterion, optimizer, num_epochs=30)

# Salvează modelul antrenat pe a doua sarcină
torch.save(model.state_dict(), './ct_model_task2.pth')

print('Antrenare completă.')
