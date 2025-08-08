import torch

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def evaluate_esmeble(models, common_loader, device):
    for model in models:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in common_loader:
            x, y = x.to(device), y.to(device)
            logits = [model(x) for model in models]
            logits = torch.stack(logits).mean(0)
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def total_evaluation(models, local_loaders, validation_loader, device):
    local_accuracies = []
    single_accuracies = []
    for model, loader in zip(models, local_loaders):
        local_accuracies.append(evaluate(model, loader, device))
        single_accuracies.append(evaluate(model, validation_loader, device))

    ensemble_accuracy = evaluate_esmeble(models, validation_loader, device)
    return {
        'ensemble_accuracy': ensemble_accuracy,
        'local_accuracies': local_accuracies,
        'single_accuracies': single_accuracies,
    }