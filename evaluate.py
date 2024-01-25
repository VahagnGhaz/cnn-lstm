import numpy as np
import torch
from tqdm import tqdm


def evaluate_model(model, val_data, loss_fn, weights=None, device='cpu', verbose=0):
    # Simplify device selection
    device = torch.device('cuda' if (device in ['gpu', 'cuda']) and torch.cuda.is_available() else 'cpu')
    model.to(device)

    if weights:
        model.load_state_dict(torch.load(weights))
        print(f'Weights loaded successfully from path: {weights}')
    
    # Evaluate the model
    model.eval()
    val_correct, running_loss = 0, 0.0
    val_total = len(val_data.dataset)

    val_iter = tqdm(val_data, desc='Evaluate', ncols=100) if verbose == 1 else val_data

    with torch.no_grad():
        for data_batch, label_batch in val_iter:
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)

            output_batch = model(data_batch)
            loss = loss_fn(output_batch, label_batch.long())
            running_loss += loss.item()

            _, predicted_labels = torch.max(output_batch.data, 1)
            val_correct += (label_batch == predicted_labels).sum().item()

    val_loss = running_loss / len(val_data)
    val_acc = val_correct / val_total
    return val_loss, val_acc

