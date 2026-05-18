import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class GraphManager():
    def __init__(self):
        pass
    
    def update_loss(self, losses, epochs):
        fig, ax = plt.subplots()
        epochs = list(range(1, epochs+1))
        ax.plot(epochs, losses, label="Train loss")
        ax.set_title("Training loss curve")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.set_xticks(epochs)
        ax.legend();
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    def update_accuracy(self, accuracies, epochs):
        fig, ax = plt.subplots()
        epochs = list(range(1, epochs+1))
        ax.plot(epochs, accuracies, label="Train accuracy")
        ax.set_title("Training accuracy curve")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_xticks(epochs)
        ax.legend();
        plt.tight_layout()
        plt.close(fig)
        return fig

    def update_confusion_matrix(self, all_labels, all_preds, class_names):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax)
        plt.tight_layout()
        return fig
        
