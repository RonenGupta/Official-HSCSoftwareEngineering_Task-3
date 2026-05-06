import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class GraphManager():
    def __init__(self, losses, epochs):
        self.losses = losses
        self.epochs = list(range(1, epochs + 1))
        
            
    def update_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.epochs, self.losses, label="Train loss")
        ax.set_title("Training loss curve")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.set_xticks(self.epochs)
        ax.legend();
        return fig

    def update_confusion_matrix(self, all_labels, all_preds, class_names):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        
