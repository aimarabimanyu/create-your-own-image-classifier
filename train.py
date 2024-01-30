import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse


def image_transforms(args):
    # Get datasets directory
    train_dir = os.path.join(args.data_directory, 'train/')
    val_dir = os.path.join(args.data_directory, 'valid/')
    
    # Train transform compose
    train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Valid transform compose
    val_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load dataset with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Transform dataset with Transform Compose respectively
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size=64)
    
    # Return class to index from train_dataset
    class_to_idx = train_datasets.class_to_idx
    
    return train_dataloaders, val_dataloaders, class_to_idx
    
def model_arch(args, class_to_idx):
    # Load pretrained model based on argument
    if args.arch == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
    elif args.arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
    elif args.arch == 'resnet34':
        model = models.resnet34(weights='DEFAULT')
    elif args.arch == 'resnet101':
        model = models.resnet101(weights='DEFAULT')
    
    # Freeze weights on pretrained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Return output from last convolution layer from pretrained model
    output_shape_pretrained_model = model.fc.in_features
    
    # Define fully-connected layer from pretrained model to suit our problem
    model.fc = nn.Sequential(nn.Linear(output_shape_pretrained_model, 128),
                             nn.ReLU(),
                             nn.Linear(128, 64),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(64, len(class_to_idx)),
                             nn.LogSoftmax(dim=1))

    # Define loss function will be used
    criterion = nn.NLLLoss()

    # Define optimizer will be used
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    return model, criterion, optimizer

def model_training(args, model, criterion, optimizer, train_dataloaders, val_dataloaders):
    # Check if want to use GPU or not
    if args.gpu:
        device = 'cuda'
        print('Model train with CUDA')
    else:
        device = 'cpu'
        print('Model train with CPU')
    
    # Move model to CUDA memory
    model.to(device)

    # Set epoch train loop
    epochs = args.epochs
    
    print('Model Training Started')
    
    # Training loop
    for e in range(epochs):
        # Initialize train loss and train accuracy
        steps = 0
        train_loss = 0
        train_accuracy = 0

        # Load training dataset
        for images, labels in train_dataloaders:
            steps += 1

            # Invoke Model Training Mode
            model.train()

            # Move Image and Label to GPU
            images, labels = images.to(device), labels.to(device)

            # Clear Optimizer
            optimizer.zero_grad()

            # Feed-Forward
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Calculate Train Accuracy
            out = torch.exp(model(images))
            top_p, top_class = out.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            # Update Train Loss & Train Accuracy
            train_loss += loss.item()
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Run validation step on end loop
            if steps == len(train_dataloaders):
                # Initialize validation loss and validation accuracy
                valid_loss = 0
                valid_accuracy = 0

                # Invoke Model Evaluation Mode
                model.eval()

                # Set No Find Gradient Descent
                with torch.no_grad():
                    for images, labels in val_dataloaders:
                        # Move Image and Label to GPU
                        images, labels = images.to(device), labels.to(device)

                        # Feed-Forward
                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)

                        # Calculate Validation Accuracy
                        out = torch.exp(model(images))
                        top_p, top_class = out.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)

                        # Update Validation Loss & Validation Accuracy
                        valid_loss += loss.item()
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print model training condition
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss/steps),
                      "Training Accuracy: {:.3f}.. ".format(train_accuracy/steps),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(val_dataloaders)),
                      "Valid Accuracy: {:.3f}.. ".format(valid_accuracy/len(val_dataloaders)))
                
    return model
                
def model_saving(args, model, class_to_idx):
    # Create directory if it doesn't exist yet
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Define checkpoint for model saving
    checkpoint = {
        'classifier' : model.fc,
        'state_dict' : model.state_dict(),
        'class_to_idx' : class_to_idx,
        'arch' : args.arch,
        'weight' : 'DEFAULT',
    }

    # Save model
    torch.save(checkpoint, os.path.join(args.save_dir, "checkpoint.pth"))
    print('Model has been saved')

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Initialize required argument
    parser.add_argument(dest='data_directory')
    
    # Initialize optional argument
    parser.add_argument('--save_dir', dest='save_dir', default='../saved_model')
    parser.add_argument('--arch', dest='arch', default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, type=float)
    parser.add_argument('--hidden_units', dest='hidden_units', default=128, type=float)
    parser.add_argument('--epochs', dest='epochs', default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    
    args = parser.parse_args()
    
    train_dataloaders, val_dataloaders, class_to_idx = image_transforms(args)
    
    model, criterion, optimizer = model_arch(args, class_to_idx)
    
    model = model_training(args, model, criterion, optimizer, train_dataloaders, val_dataloaders)
               
    model_saving(args, model, class_to_idx)

if __name__ == "__main__":
    main()