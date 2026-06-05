import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

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
        plt.close(fig)
        return fig
    
    def update_gpu_plot(self, gpu_history):
        fig, ax = plt.subplots()
        ax.plot(gpu_history, color="red")
        ax.set_ylim(0, 100)
        ax.set_title("GPU Load (%)")
        ax.set_xlabel("Update step")
        ax.set_ylabel("Load %")
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    def update_cpu_ram_plot(self, cpu_history, ram_history):
        fig, ax = plt.subplots()
        ax.plot(cpu_history, label="CPU", color="blue")
        ax.plot(ram_history, label="RAM", color="green")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Update step")
        ax.set_ylabel("Usage %")
        ax.legend()
        ax.set_title("CPU / RAM (%)")
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    def rebuild_confusion_matrix(self, stored, class_names):
        labels, preds = stored
        labels = np.array(labels, dtype=int)
        preds = np.array(preds, dtype=int)

        num_classes = len(class_names)
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax)
        plt.tight_layout()
        plt.close(fig)
        return fig