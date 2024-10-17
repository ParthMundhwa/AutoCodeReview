import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir, batch_size):
    """Load training and test datasets with transformations."""
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def save_model(model, path):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load a model from a file."""
    model.load_state_dict(torch.load(path))
    return model