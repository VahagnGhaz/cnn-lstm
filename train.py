import torch
from tqdm import tqdm
from evaluate import evaluate_model

def train_model(model, train_data, loss_fn, optimizer, epochs, device, val_data=None, save_best_path=None):
    # Move model to device
    model.to(device)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0.0, 0
        # Training loop
        for inputs, labels in tqdm(train_data, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_data)
        train_acc = total_correct / (len(train_data.dataset))

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # Validation loop
        if val_data:
            model.eval()
            val_loss, val_acc = evaluate_model(model, val_data, loss_fn, device=device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if save_best_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_best_path)

        # Print epoch results
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return model, history
