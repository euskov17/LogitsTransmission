import random
import torch
import numpy as np

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

def feen_training_step(models, states, optimizers, local_loaders, common_loader, criterion, device, n_global_steps=3, n_local_steps=3):
    for _ in range(n_global_steps):
        batch = next(iter(common_loader))
        sync_clients(models, criterion, batch, device)

        for model, state in zip(models, states):
            state.global_step(model)
    
    loss_history = []
    loss_reg_history = []
    for model, state, loader, optimizer in zip(models, states, local_loaders, optimizers):
        state.set_weights(model)

        for param, new_param in zip(model.parameters(), state.state):
            param.data.add_(-state.gamma * state.lmbd * new_param)

        for _ in range(n_local_steps):
            inputs, target = next(iter(loader))
            inputs = inputs.to(device)
            target = target.to(device)
            logits = model(inputs)
            loss_local = criterion(logits, target)

            reg = state.get_reg_term(model)
            loss = loss_local * state.gamma + reg
            # loss *= 1000

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(state.gamma * loss_local.detach().cpu().item())
            loss_reg_history.append(reg.detach().cpu().item())
    
    return np.mean(loss_history), np.mean(loss_reg_history) 
    # import numpy as np
    # print(f"Local loss avg = {np.mean(loss_history)}, Reg loss part {np.mean(loss_reg_history)}")
