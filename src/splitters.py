from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import torch
import random

def get_client_loaders(num_clients=3, train=True):
    transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
    )
    mnist_train = datasets.CIFAR10(root='./data_cifar10', train=train, download=True, transform=transform)
    client_data = []
    samples_per_client = len(mnist_train) // num_clients
    for i in range(num_clients):
        indices = range(i * samples_per_client, (i+1) * samples_per_client)
        client_data.append(Subset(mnist_train, indices))
    
    client_loaders = [DataLoader(data, batch_size=32, shuffle=True, drop_last=True) for data in client_data]
    return client_loaders

def get_imbalanced(num_clients=12):
    transform = transforms.ToTensor()
    mnist_train = datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
    samples_per_client = len(mnist_train) // num_clients
    common_data, validation_data = Subset(mnist_train, range(0 * samples_per_client, (0+1) * samples_per_client)), Subset(mnist_train, range(1 * samples_per_client, (1+1) * samples_per_client))
    clients_dataset = Subset(mnist_train, range(2 * samples_per_client, (num_clients+1) * samples_per_client))

    labels = torch.tensor([label for _, label in clients_dataset])
    sorted_indices = torch.argsort(labels)
    sorted_dataset = Subset(clients_dataset, sorted_indices)
    client_data = []
    
    for i in range(num_clients - 2):
        indices = range(i * samples_per_client, (i+1) * samples_per_client)
        client_data.append(Subset(sorted_dataset, indices))

    client_data.append(common_data)
    client_data.append(validation_data)
    client_loaders = [DataLoader(data, batch_size=32, shuffle=True, drop_last=True) for data in client_data]
    return client_loaders

def create_probabilistic_client_loaders(
    num_clients=20,
    p=0.7,  # Probability a client gets a class
    batch_size=32,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    dataset = datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
    
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(10):
        # Determine which clients get this class
        client_mask = np.random.rand(num_clients) < p
        selected_clients = np.where(client_mask)[0]
        
        if len(selected_clients) > 0:
            splits = np.array_split(class_indices[class_idx], len(selected_clients))
            for client_id, split in zip(selected_clients, splits):
                client_indices[client_id].extend(split)
    
    # Create DataLoaders
    client_loaders = []
    for indices in client_indices:
        if not indices:
            indices = [0]
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        client_loaders.append(loader)
    
    return client_loaders

def create_probabilistic_client_loaders(
    num_clients=20,
    p=0.7,  # Probability a client gets a class
    batch_size=32,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(10):
        # Determine which clients get this class
        client_mask = np.random.rand(num_clients) < p
        selected_clients = np.where(client_mask)[0]
        
        if len(selected_clients) > 0:
            # Split class indices among selected clients
            splits = np.array_split(class_indices[class_idx], len(selected_clients))
            for client_id, split in zip(selected_clients, splits):
                client_indices[client_id].extend(split)
    
    # Create DataLoaders
    client_loaders = []
    for indices in client_indices:
        if not indices:
            indices = [0]
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        client_loaders.append(loader)
    
    return client_loaders

def get_imbalanced_client_loaders(num_clients=3, classes_per_client=7, batch_size=128, class_elements_per_user=500):
    client_train_loaders = []
    client_val_loaders = []
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    cifar_train = datasets.CIFAR10(root='./data_cifar', train=True,
                                   download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./data_cifar', train=False,
                                   download=True, transform=transform)
    cifar_common = Subset(cifar_train, range(1000))
    cifar_train = Subset(cifar_train, range(1000, len(cifar_train)))
    
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(cifar_train):
        class_indices[label].append(idx)
    for i in range(num_clients):
        client_indices = []
        client_classes = random.sample(range(10), classes_per_client)
        for obj_class in client_classes:
            client_indices += random.sample(class_indices[obj_class], class_elements_per_user)
        data = Subset(cifar_train, client_indices)
        train_size = len(data) - 1000
        val_size = 1000
        train, val = random_split(data, [train_size, val_size])
        train = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
        val = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
        client_train_loaders.append(train)
        client_val_loaders.append(val)
    
    # client_loaders = [DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False) for data in client_data]
    common_loader = DataLoader(cifar_common, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, drop_last=False)
    return client_train_loaders, client_val_loaders, common_loader, validation_loader



def get_imbalanced_client_loaders_cifar100(num_clients=3, classes_per_client=30, batch_size=128, class_elements_per_user=50):
    client_train_loaders = []
    client_val_loaders = []
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    cifar_train = datasets.CIFAR100(root='./data_cifar100', train=True,
                                   download=True, transform=transform)
    cifar_test = datasets.CIFAR100(root='./data_cifar100', train=False,
                                   download=True, transform=transform)
    cifar_common = Subset(cifar_train, range(5000))
    cifar_train = Subset(cifar_train, range(5000, len(cifar_train)))
    
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(cifar_train):
        class_indices[label].append(idx)
    for i in range(num_clients):
        client_indices = []
        client_classes = random.sample(range(100), classes_per_client)
        for obj_class in client_classes:
            client_indices += random.sample(class_indices[obj_class], class_elements_per_user)
        data = Subset(cifar_train, client_indices)
        train_size = len(data) - 1000
        val_size = 1000
        train, val = random_split(data, [train_size, val_size])
        train = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
        val = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
        client_train_loaders.append(train)
        client_val_loaders.append(val)
    
    # client_loaders = [DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False) for data in client_data]
    common_loader = DataLoader(cifar_common, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, drop_last=False)
    return client_train_loaders, client_val_loaders, common_loader, validation_loader