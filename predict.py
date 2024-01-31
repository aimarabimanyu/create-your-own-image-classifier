import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import json

def load_checkpoint(args):
    # Load checkpoint file
    checkpoint = torch.load(args.checkpoint_path)
    
    # Load whatever is needed for the model
    model = getattr(models, checkpoint['arch'])(weights = checkpoint['weight'])
    if checkpoint['arch'] in ['vgg11', 'vgg13', 'vgg16']:
        model.classifier = checkpoint['classifier']
    else:
        model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Freeze weights on loaded model
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    
def load_categories_name(args):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name
    
def process_image(args):
    image = Image.open(args.input_image_path)

    # Transform Compose for loaded image
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
    ])

    # Transform loaded images
    image = image_transforms(image)

    return image
    
def predict(args, model, image, cat_to_name):
    # Check if want to use GPU or not
    if args.gpu:
        device = 'cuda'
        print('Predict with CUDA')
    else:
        device = 'cpu'
        print('Predict with CPU')
        
    # Move model to CUDA memory
    model.to(device)

    # Invoke Model Evaluation Mode
    model.eval()
    
    image = image.unsqueeze(0)

    # Set No Find Gradient Descent
    with torch.no_grad():
        # Move Image and Label to GPU
        image = image.to(device)

        # Feed-Forward
        log_ps = model.forward(image)

    # Calculate probabilities
    out_prob = torch.exp(log_ps)

    # Find top class and probabilities
    top_p, top_class = out_prob.topk(args.top_k)

    # Change top probabilities to list look
    top_p = top_p[0].tolist()

    # Load index to class dictionary
    index_to_class = {val: key for key, val in model.class_to_idx.items()}

    # Arrange top class based on index_to_class
    top_class = [index_to_class[key] for key in top_class[0].tolist()]
    
    # Arrange top class based on class to name
    top_class = [cat_to_name[str(key)] for key in top_class]
    
    for i in range(args.top_k):
        print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(top_class[i], 100 * top_p[i]))
    
def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Initialize required argument
    parser.add_argument(dest='input_image_path')
    parser.add_argument(dest='checkpoint_path')
    
    # Initialize optional argument
    parser.add_argument('--top_k', dest='top_k', default='5', type=int)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    
    args = parser.parse_args()
    
    model = load_checkpoint(args)
    
    image = process_image(args)
    
    cat_to_name = load_categories_name(args)
    
    predict(args, model, image, cat_to_name)
    
if __name__ == "__main__":
    main()