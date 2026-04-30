import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
matplotlib.use('MacOSX')

class GraphManager():
    def __init__(self, losses, epochs):
        self.losses = losses
        self.epochs = list(range(1, epochs + 1))
        
            
    def update_loss(self):
        plt.plot(self.epochs, self.losses, label="Train loss")
        plt.title("Training loss curve")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.xticks(self.epochs)
        plt.legend();
        plt.show()

    def update_confusion_matrix(self, all_labels, all_preds, class_names):
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.show()
