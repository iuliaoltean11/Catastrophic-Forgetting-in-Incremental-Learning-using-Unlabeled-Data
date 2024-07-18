import torch
from PIL import Image
import torchvision.transforms as transforms
from model_pt import SimpleCNN  # Import the SimpleCNN model Pt

def predict(model, image, device):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_path = '' # schimbati cu calea imaginilor.
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=5)  # Adjust num_classes for the subset
    model.load_state_dict(torch.load('./pt_model.pth'))
    model.to(device)

    prediction = predict(model, image, device)
    print(f'Predicted class: {prediction.item()}')