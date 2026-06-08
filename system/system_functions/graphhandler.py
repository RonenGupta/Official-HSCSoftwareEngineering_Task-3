import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gradio as gr
import numpy as np

class GraphManager():
    """Handles graphing for loss and accuracy curves, confusion matrix in PDF and testing, CPU/RAM and GPU plots"""
    def __init__(self):
        # Does not take any initialisation states
        pass
    
    def update_loss(self, losses, epochs):
        """Handles loss curves, takes in loss and epoch data and returns the fig plot"""
        # Check if epoch and loss data is valid for plotting
        if not losses or epochs <= 0 or len(losses) != epochs:
            return gr.Warning("Invalid loss data for plotting.")
        try:
            # Create loss graph for training, title, plot X and Y and add a legend
            fig, ax = plt.subplots()
            epochs = list(range(1, epochs+1))
            ax.plot(epochs, losses, label="Train loss")
            ax.set_title("Training loss curve")
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epochs")
            ax.set_xticks(epochs)
            ax.legend();
            # Adjusts spacing so labels, titles, tick marks dont overlap and close graph to 
            # prevent it from staying in memory, return plot
            plt.tight_layout()
            plt.close(fig)
            return fig
        # Error handling if plot was not generated for another reason
        except Exception as e:
            return gr.Warning(f"Error generating loss plot: {str(e)}")
    
    def update_accuracy(self, accuracies, epochs):
        """Handles accuracy curves, takes in accuracy and epoch data and returns the fig plot"""
        # Check if epoch and accuracy data is valid for plotting
        if not accuracies or epochs <= 0 or len(accuracies) != epochs:
            return gr.Warning("Invalida ccuracy data for plotting.")
        try:
            # Create accuracy graph for training, title, plot X and Y and add a legend
            fig, ax = plt.subplots()
            epochs = list(range(1, epochs+1))
            ax.plot(epochs, accuracies, label="Train accuracy")
            ax.set_title("Training accuracy curve")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Epochs")
            ax.set_xticks(epochs)
            ax.legend();
            # Adjusts spacing so labels, titles, tick marks dont overlap and close graph to 
            # prevent it from staying in memory, return plot
            plt.tight_layout()
            plt.close(fig)
            return fig
        # Error handling if plot was not generated for another reason
        except Exception as e:
            return gr.Warning(f"Error generating accuracy plot: {str(e)}")

    def update_confusion_matrix(self, all_labels, all_preds, class_names):
        """Handles confusion matrix in testing, takes in all predicted
         labels and all true labels and class names and returns the fig plot"""
        # Check if label and pred data is valid for plotting
        if not all_labels or not all_preds or len(all_labels) != len(all_preds):
            return gr.Warning("Invalid labels or predictions for confusion matrix.")
        try:
            # Use sklearn built in method and class for confusion matrix display, requiring class names
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax)
            # Adjusts spacing so labels, titles, tick marks dont overlap and close graph to 
            # prevent it from staying in memory, return plot
            plt.tight_layout()
            plt.close(fig)
            return fig
        # Error handling if plot was not generated for another reason
        except Exception as e:
            return gr.Warning(f"Error generating confusion matrix: {str(e)}")
    
    def update_gpu_plot(self, gpu_history):
        """Handles GPU plots in training, takes in GPU history data
        and returns plot"""
        # Check if GPU history is available
        if not gpu_history:
            return gr.Warning("No GPU history data for plotting.")
        try:
            fig, ax = plt.subplots()
            ax.plot(gpu_history, color="red")
            ax.set_ylim(0, 100)
            ax.set_title("GPU Load (%)")
            ax.set_xlabel("Update step")
            ax.set_ylabel("Load %")
            # Adjusts spacing so labels, titles, tick marks dont overlap and close graph to 
            # prevent it from staying in memory, return plot
            plt.tight_layout()
            plt.close(fig)
            return fig
        # Error handling if plot was not generated for another reason
        except Exception as e:
            return gr.Warning(f"Error generating GPU load plot: {str(e)}")
    
    def update_cpu_ram_plot(self, cpu_history, ram_history):
        """Handles CPU/RAM plots in training, takes in CPU/RAM history data
        and returns plot"""
        # Checks if CPU and RAM history are available and equal in length for plotting
        if not cpu_history or not ram_history or len(cpu_history) != len(ram_history):
            return gr.Warning("Invalid CPU or RAM history data for plotting.")
        try:
            fig, ax = plt.subplots()
            # This time, sets two plots, one for CPU and one for RAM and set
            # y limit of 100 to facilitate percentage usage
            ax.plot(cpu_history, label="CPU", color="blue")
            ax.plot(ram_history, label="RAM", color="green")
            ax.set_ylim(0, 100)
            ax.set_xlabel("Update step")
            ax.set_ylabel("Usage %")
            ax.legend()
            ax.set_title("CPU / RAM (%)")
            # Adjusts spacing so labels, titles, tick marks dont overlap and close graph to 
            # prevent it from staying in memory, return plot
            plt.tight_layout()
            plt.close(fig)
            return fig
        # Error handling if plot was not generated for another reason
        except Exception as e:
            return gr.Warning(f"Failed to plot CPU/RAM usage: {e}")
    
    def rebuild_confusion_matrix(self, stored, class_names):
        """Handles confusion matrix in PDF generation, takes in stored confusion matrix entries
        in the form of integers and class names and returns plot"""
        try:
            # Convert labels and preds to numpy arrays from stored entries
            labels, preds = stored
            labels = np.array(labels, dtype=int)
            preds = np.array(preds, dtype=int)
            num_classes = len(class_names)
            # Plot confusion matrix using the same sklearn method, taking in labels, preds,
            # and class names, return plot
            cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax)
            plt.tight_layout()
            plt.close(fig)
            return fig
        # Error handling if plot was not generated for another reason
        except Exception as e:
            return gr.Warning(f"Failed to rebuild confusion matrix in PDF: {str(e)}")