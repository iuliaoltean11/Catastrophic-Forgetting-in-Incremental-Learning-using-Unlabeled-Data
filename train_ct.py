import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model_ct import CTNet  # Ensure this is your CT model
from model_pt import SimpleCNN
def filter_and_transform_classes(dataset, classes):
    indices = [i for i, target in enumerate(dataset.targets) if target in classes]
    dataset.targets = [dataset.targets[i] - classes[0] for i in indices]  # Adjust targets to start from 0
    dataset.data = dataset.data[indices]
    return dataset

def train(model, teacher_model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data).detach()
        loss = distillation_loss(output, target, teacher_output, T=2.0, alpha=0.7)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return train_loss / len(train_loader.dataset)

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)')
    return val_loss, val_accuracy

def distillation_loss(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(nn.functional.log_softmax(y / T, dim=1),
                          nn.functional.softmax(teacher_scores / T, dim=1)) * (alpha * T * T) + \
           nn.CrossEntropyLoss()(y, labels) * (1. - alpha)

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainset = filter_and_transform_classes(trainset, classes=[5, 6, 7, 8, 9])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    valset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    valset = filter_and_transform_classes(valset, classes=[5, 6, 7, 8, 9])
    val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    teacher_model = SimpleCNN().to(device)
    teacher_model.load_state_dict(torch.load("pt_model.pth"))
    teacher_model.eval()

    model = CTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, 51):
        train_loss = train(model, teacher_model, device, train_loader, optimizer, epoch)
        val_loss, val_accuracy = validate(model, device, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    torch.save(model.state_dict(), "model_ct.pth")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.show()

if __name__ == '__main__':
    main()
