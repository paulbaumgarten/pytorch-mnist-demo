# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


"""
# From https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/10
pil_img = Image.open(img)
print(pil_img.size)  
 
pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
print(pil_to_tensor.shape) 

tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
print(tensor_to_pil.size)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import os
from PIL import Image

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "9", "9", "0"]
model.eval()

images = ["5-bw.png", "4-bw.png", "3-bw.png", "0.png", "1.png", "2.png", "6.png", "7.png", "8.png", "9.png", "9b.png"]
for img in images:
    print("Filename: ",img)
    tester = transforms.ToTensor()(Image.open(img))
    with torch.no_grad():
        pred = model(tester)
        print(" --> ",pred[0])
        best = pred[0].argmax(0)
        print(f"Prediction is { classes[best] }")

# I imagine the next step is to use your own datasets. This looks ussful
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
