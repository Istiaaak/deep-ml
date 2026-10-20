# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    history = []
    
    # Your code here
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        indices = torch.randperm(X_train.shape[0])
        model.train()
        total_train_loss = 0
        num_batches = math.ceil(X_train.shape[0]/batch_size)
        for start in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[start: start+batch_size]
            optimizer.zero_grad()
            pred = model(X_train[batch_idx])
            loss = loss_fn(pred, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()
        train_loss = total_train_loss / num_batches
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)
            val_cls = val_pred.argmax(dim=1)
            
            val_acc = (val_cls == y_val).float().mean().item()
        
        history.append({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss.item(),
        "val_accuracy": val_acc
        })
        
    return history