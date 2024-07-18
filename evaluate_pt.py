import torch
from torchvision import datasets, transforms
from model_pt import SimpleCNN  # Import the SimpleCNN model Pt


def evaluate_pt_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes_to_keep = [0, 1, 2, 3, 4]
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(classes_to_keep)).to(device)
    model.load_state_dict(torch.load('./pt_model.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            mask = (labels < 5)  # Filter for specific classes
            inputs, labels = inputs[mask], labels[mask]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the Pt model on the test images: {100 * correct / total}%')


if __name__ == "__main__":
    evaluate_pt_model()
