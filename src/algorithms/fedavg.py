import torch

def fed_avg_training_step(models, optimizers, local_loaders, criterion, device, n_local_steps=3):
    for model, optimizer, loader in zip(models, optimizers, local_loaders):
        model.train()
        for _ in range(n_local_steps):
            inputs, target = next(iter(loader))
            inputs = inputs.to(device)
            target = target.to(device)
            logits = model(inputs)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    avg_model = [torch.zeros_like(param) for param in models[0].parameters()]

    for model in models:
        for idx, param in enumerate(model.parameters()):
            avg_model[idx] += param
    
    for model in models:
        for param, new_param in zip(model.parameters(), avg_model):
            param.data.copy_(new_param / len(models))
