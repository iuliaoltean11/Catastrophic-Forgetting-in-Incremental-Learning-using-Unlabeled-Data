import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model_pt import SimpleCNN  # Import the SimpleCNN model Pt

"""
    Rol: Filtrează dataset-ul pentru a păstra doar anumite clase specificate.
    targets: Convertește țintele dataset-ului într-un tensor PyTorch. Creează o mască booleană inițială cu toate valorile false.
    Actualizează dataset-ul pentru a păstra doar datele și țintele filtrate.
"""
def filter_classes(dataset, classes):
    targets = torch.tensor(dataset.targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for cls in classes:
        mask = mask | (targets == cls)
    dataset.targets = targets[mask].tolist()
    dataset.data = dataset.data[mask.numpy()]
    return dataset

"""
    Transformare in tensori și normalizare.
    Creează un loader de date pentru a itera prin dataset în batch-uri de 100, shuffle fiind setat pentru a amesteca datele.
"""
def train_pt_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes_to_keep = [0, 1, 2, 3, 4]  # Subset of classes
    trainset = filter_classes(trainset, classes_to_keep)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(classes_to_keep)).to(device)
    # Funcția de pierdere Cross-Entropy.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data # despachetare date
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #update gradient
            outputs = model(inputs) # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), './pt_model.pth')

if __name__ == "__main__":
    train_pt_model()
