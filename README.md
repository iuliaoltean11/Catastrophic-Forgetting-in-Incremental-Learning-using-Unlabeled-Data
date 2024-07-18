To use this project, we have to:
# Train the teacher model (SimpleCNN)
python train_pt.py

# Train the student model (CTNet) with knowledge distillation
python train_ct.py

# Evaluate the student model (CTNet)
python test_ct.py

# Predict custom images using the trained student model (CTNet)
python predict_custom_images_ct.py

Important Notes!
The student model is specifically trained to recognize CIFAR-100 classes 5 through 9.
The scripts assume the CIFAR-100 dataset will be automatically downloaded and stored in the ./data directory.
Ensure all paths and dependencies are correctly set up before running the scripts.
For the CT model, there are 2 options implemented, and the "predict_" files are used for the specific classifications when a chosen image is uploaded.
