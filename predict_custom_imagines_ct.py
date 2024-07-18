import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model_ct import CTNet  # Importing the CTNet model from model_ct.py

# CIFAR-100 classes 5-9
class_names = ['dog', 'frog', 'horse', 'ship', 'truck']

# Function to load the model
def load_model(model_path, device):
    model = CTNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess and load custom images
def load_custom_images(image_path, transform, device):
    images = []
    image_names = []
    if os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_path, filename)
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                images.append(image)
                image_names.append(filename)
    elif os.path.isfile(image_path) and image_path.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        images.append(image)
        image_names.append(os.path.basename(image_path))
    return images, image_names

# Function to make predictions on custom images
def predict_custom_images(model, device, images, image_names):
    with torch.no_grad():
        for image, name in zip(images, image_names):
            output = model(image)
            probabilities = nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_class = predicted.item()
            print(f'Image: {name}, Predicted Class: {predicted_class} ({class_names[predicted_class]}), Probabilities: {probabilities.cpu().numpy()}')

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = "model_ct.pth"
    model = load_model(model_path, device)

    # Define the transform to match the training preprocess
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize image to 32x32 pixels
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Path to your custom image or directory containing images
    image_path = r"D:\Python_Projects_2024_PyCharm\GoodCatastrophic_Forgetting\GoodCatastrophic_Forgetting\pictures"

    # Load and preprocess custom images
    images, image_names = load_custom_images(image_path, transform, device)

    # Ensure images are correctly loaded
    if not images:
        print("No images loaded. Please check the image directory and file formats.")
        return

    # Make predictions on custom images
    predict_custom_images(model, device, images, image_names)

if __name__ == '__main__':
    main()
