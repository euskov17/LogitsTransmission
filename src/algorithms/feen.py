import random
import torch

def sync_clients(models, criterion, batch, device):
    inputs, target = batch
    inputs = inputs.to(device)
    target = target.to(device)
    logits = []
    for model in models:
        model.train()
        logits.append(model(inputs))

    logits = torch.stack(logits)
    logits = logits.mean(0)
    loss = criterion(logits, target)
    loss.backward()

def feen_training_step(models, states, optimizers, local_loaders, common_loader, criterion, device, n_local_steps=3):
    batch = next(iter(common_loader))
    sync_clients(models, criterion, batch, device)

    for model, state in zip(models, states):
        state.global_step(model)
    
    for model, state, loader, optimizer in zip(models, states, local_loaders, optimizers):
        state.set_weights(model)
        for _ in range(n_local_steps):
            inputs, target = next(iter(loader))
            inputs = inputs.to(device)
            target = target.to(device)
            logits = model(inputs)
            loss = criterion(logits, target)

            reg = state.get_reg_term(model)
            loss = loss * state.gamma + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()