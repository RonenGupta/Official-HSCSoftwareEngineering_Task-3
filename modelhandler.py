import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import json  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device

USERS_JSON = Path("users.json")

class ModelManager():
    def __init__(self):
        pass

    def test(self, model):
        """Testing loop"""
        loss_fn = torch.nn.CrossEntropyLoss()
        test_loss = 0
        test_acc = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(device)
                y = y.to(device)

                y_pred = model(X)
                _, preds = torch.max(y_pred, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                loss = loss_fn(y_pred, y)

                test_loss += loss.item()
                test_acc += accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())

            avg_test_loss = test_loss / len(self.test_dataloader)
            avg_test_acc = test_acc / len(self.test_dataloader)

            test_metrics = f"Test Loss: {avg_test_loss} || Test Accuracy: {avg_test_acc}\n Test Precision: {precision_score(all_labels, all_preds, average='macro')}\nTest Recall: {recall_score(all_labels, all_preds, average='macro')}\nTest F1-Score: {f1_score(all_labels, all_preds, average='macro')}"

        return test_metrics, all_labels, all_preds

    def train(self, epochs: int, lr: float = 0.01):
        """Training loop"""
        log = ""
        param_list = []

        for param in self.model.parameters():
            if param.requires_grad:
                param_list.append(param)

        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(param_list, lr)

        EPOCHS = epochs
        losses = []
        for epoch in range(EPOCHS):
            train_loss = 0
            train_acc = 0
            
            self.model.train()
            for X, y in self.train_dataloader:
                X = X.to(device)
                y = y.to(device)

                y_pred = self.model(X)

                loss = loss_fn(y_pred, y)

                train_loss += loss.item()
                train_acc += accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_train_acc = train_acc / len(self.train_dataloader)
            
            log += f"Epoch: {epoch + 1} || Train Loss: {avg_train_loss} || Train Accuracy {avg_train_acc}\n"
            losses.append(avg_train_loss)
            
            yield log
        return losses

    def build(self, layer4: bool = False):
        """Model build process"""
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        self.model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=512,
                    out_features=len(self.train_dataset.classes),
                    bias=True)).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        if layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
    
    def test_transforms_dataset(self, test_transforms: transforms.Compose | None, test_path: str, test_bs: int = 32):
        """Setup dataloader for testing data"""
        default_test_transform = transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std=[0.229, 0.224, 0.225])
                                            ])
        if test_transforms == None:
            self.test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform = default_test_transform)
        else:
            self.test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform = test_transforms)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=test_bs, shuffle=True)

        return self.test_dataset.classes
    
    def train_transforms_dataset(self, train_transforms: transforms.Compose | None, train_path: str, train_bs: int = 32,):
        """Setup dataloader for training data"""
        default_train_transform = transforms.Compose([
                                              transforms.Resize((256, 256)),
                                              transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                   std=[0.229, 0.224, 0.225])
                                              ])
        if train_transforms == None:
            self.train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform = default_train_transform)
        else:
            self.train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform = train_transforms)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True)

    def save_model(self, model, username, model_name):

        with open(USERS_JSON, "r") as f:
            users = json.load(f)
        
        if username not in users:
            return "User not found"
        
        models = users[username].get("models", {})

        if model_name in models:
            model_path = Path(models[model_name])

        else:

            model_path = Path(f"saved_models/{username}_{model_name}.pth")

            models[model_name] = str(model_path)
            users[username]["models"] = models

            with open (USERS_JSON, "w") as f:
                json.dump(users, f, indent=4)

        torch.save(model.state_dict(), model_path)

        return f"Model '{model_name}' saved!"

    def load_model(self, username, model_name, layer4, num_classes = 2):
        with open(USERS_JSON, 'r') as f:
            users = json.load(f)
        user_models = users[username].get("models", {})
        model_path = user_models[model_name]
        self.model = torchvision.models.resnet18(weights=None).to(device)
        self.model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(512, num_classes)).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        if layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        return self.model