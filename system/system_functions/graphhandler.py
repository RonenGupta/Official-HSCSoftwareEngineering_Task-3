import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gradio as gr
import numpy as np

class GraphManager():
    def __init__(self):
        pass
    
    def update_loss(self, losses, epochs):
        if not losses or epochs <= 0 or len(losses) != epochs:
            return gr.Warning("Invalid loss data for plotting.")
        try:
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
        except Exception as e:
            return gr.Warning(f"Error generating loss plot: {str(e)}")
    
    def update_accuracy(self, accuracies, epochs):
        if not accuracies or epochs <= 0 or len(accuracies) != epochs:
            return gr.Warning("Invalida ccuracy data for plotting.")
        try:
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
        except Exception as e:
            return gr.Warning(f"Error generating accuracy plot: {str(e)}")

    def update_confusion_matrix(self, all_labels, all_preds, class_names):
        if not all_labels or not all_preds or len(all_labels) != len(all_preds):
            return gr.Warning("Invalid labels or predictions for confusion matrix.")
        try:
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax)
            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            return gr.Warning(f"Error generating confusion matrix: {str(e)}")
    
    def update_gpu_plot(self, gpu_history):
        if not gpu_history:
            return gr.Warning("No GPU history data for plotting.")
        try:
            fig, ax = plt.subplots()
            ax.plot(gpu_history, color="red")
            ax.set_ylim(0, 100)
            ax.set_title("GPU Load (%)")
            ax.set_xlabel("Update step")
            ax.set_ylabel("Load %")
            plt.tight_layout()
            plt.close(fig)
            return fig
        except Exception as e:
            return gr.Warning(f"Error generating GPU load plot: {str(e)}")
    
    def update_cpu_ram_plot(self, cpu_history, ram_history):
        if not cpu_history or not ram_history or len(cpu_history) != len(ram_history):
            return gr.Warning("Invalid CPU or RAM history data for plotting.")
        try:
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
        except Exception as e:
            return gr.Warning(f"Failed to plot CPU/RAM usage: {e}")
    
    def rebuild_confusion_matrix(self, stored, class_names):
        try:
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
        except Exception as e:
            return gr.Warning(f"Failed to rebuild confusion matrix in PDF: {str(e)}")