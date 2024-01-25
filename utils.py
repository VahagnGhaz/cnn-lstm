import json
import matplotlib.pyplot as plt

def load_params(file_path):
    with open(file_path, 'r') as file:
        params = json.load(file)

    return params

def visualize_history(history, metrics=['acc', 'loss'], save_path=None):
    plt.figure(figsize=(12, 6))

    if 'acc' in metrics:
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='train_acc', c='r')
        plt.plot(history['val_acc'], label='val_acc', c='g')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    if 'loss' in metrics:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='train_loss', c='r')
        plt.plot(history['val_loss'], label='val_loss', c='g')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()
