import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model_ct import CTNet  # Importing the CTNet model from model_ct.py


def load_model(model_path, device):
    model = CTNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def get_test_loader(classes, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Filter the dataset to only include classes 5-9 and adjust labels
    indices = [i for i, target in enumerate(testset.targets) if target in classes]
    testset.targets = [testset.targets[i] - classes[0] for i in indices]  # Adjust targets to start from 0
    testset.data = testset.data[indices]

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


def predict(model, device, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = "model_ct.pth"
    model = load_model(model_path, device)

    classes = range(5, 10)
    test_loader = get_test_loader(classes)

    predict(model, device, test_loader)


if __name__ == '__main__':
    main()
