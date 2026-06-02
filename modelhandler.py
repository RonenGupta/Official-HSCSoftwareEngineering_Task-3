import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
from PIL import Image
import numpy as np
import json
import math
import time
import os  
import pickle
import datetime
import psutil
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

    def train(self, earlystopping, patience, epochs: int, lr: float = 0.01):
        """Training loop"""
        log = ""
        param_list = []

        for param in self.model.parameters():
            if param.requires_grad:
                param_list.append(param)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(param_list, lr)

        best_loss = float('inf')
        early_stop_counter = 0

        EPOCHS = epochs
        losses = []
        accuracies= []

        for epoch in range(EPOCHS):
            train_loss = 0
            train_acc = 0
            
            self.model.train()
            for X, y in self.train_dataloader:

                step_start = time.time()

                X = X.to(device)
                y = y.to(device)

                y_pred = self.model(X)

                loss = loss_fn(y_pred, y)

                train_loss += loss.item()
                train_acc += accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                step_time = time.time() - step_start
                iters_per_sec = 1 / step_time

                cpu, ram = self.get_cpu_ram_usage()
                gpu = self.get_gpu_usage()

                analytics = {
                    "epoch": epoch + 1,
                    "step_time": step_time,
                    "iters_per_sec": iters_per_sec,
                    "cpu": cpu,
                    "ram": ram,
                    "gpu": gpu
                }

            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_train_acc = train_acc / len(self.train_dataloader)
            
            losses.append(avg_train_loss)
            accuracies.append(avg_train_acc)

            log += f"Epoch: {epoch + 1} || Train Loss: {avg_train_loss} || Train Accuracy {avg_train_acc}\n"

            if earlystopping:
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    early_stop_counter = 0
                    log += "Best loss improved"
                else:
                    early_stop_counter += 1
                    log += f"No improvement ({early_stop_counter/patience})"
                    
                    if early_stop_counter >= patience:
                        log += f"\nEarly stopping triggered at epoch {epoch+1}!"
                        break
            log += "\n"

            yield log, losses, accuracies, analytics
        return losses, accuracies

    def build(self, architecture: str = "ResNet18", layer1: bool = False, layer2: bool = False, layer3: bool = False, layer4: bool = False, dropout=0.2):
        """Model build process"""
        resnet_models = {
            "ResNet18": (torchvision.models.resnet18, 512),
            "ResNet34": (torchvision.models.resnet34, 512),
            "ResNet50": (torchvision.models.resnet50, 2048),
            "ResNet101": (torchvision.models.resnet101, 2048),
            "ResNet152": (torchvision.models.resnet152, 2048),
        }
        model_arch, in_features = resnet_models[architecture]
        self.model = model_arch(weights="DEFAULT").to(device)
        self.model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout, inplace=True),
        torch.nn.Linear(in_features=in_features,
                    out_features=len(self.train_dataset.classes),
                    bias=True)).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

        layer_map = {
            layer1: "layer1",
            layer2: "layer2",
            layer3: "layer3",
            layer4: "layer4"
        }

        for name, key in layer_map.items():
            if name:
                layer = getattr(self.model, key)
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.layer_config = {
            "layer1": layer1,  
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4     
        }
    
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

    def save_model(self, model, username, model_name, final_accuracy, final_loss, epochs, loss_curve, accuracy_curve, arch_type):

        with open(USERS_JSON, "r") as f:
            users = json.load(f)
        
        if username not in users:
            return "User not found"
        
        models = users[username].get("models", {})

        if model_name in models:
            model_path = Path(models[model_name]["path"])

        else:

            model_path = Path(f"saved_models/{username}_{model_name}.pth")

            models[model_name] = {
                "path": str(model_path),
                "accuracy": float(final_accuracy),
                "loss": float(final_loss),
                "epochs": int(epochs),
                "date": str(datetime.datetime.now()),
                "architecture": arch_type,
                "loss_curve": [float(x) for x in loss_curve],
                "accuracy_curve": [float(x) for x in accuracy_curve],
                "confusion_matrix": None,
                "layers": self.layer_config,
                "class_names": None,
            }

            models[model_name]["notes"] = "My honest reaction:"
            users[username]["models"] = models

            with open (USERS_JSON, "w") as f:
                json.dump(users, f, indent=4)

        torch.save(model.state_dict(), model_path)

        return f"Model '{model_name}' saved!"

    def load_model(self, username, model_name, num_classes = 2):
        with open(USERS_JSON, 'r') as f:
            users = json.load(f)
        user_models = users[username].get("models", {})
        model_path = user_models[model_name]["path"]
        model_arch = user_models[model_name]["architecture"]
        layer_config = user_models[model_name]["layers"]

        resnet_models = {
            "ResNet18": (torchvision.models.resnet18, 512),
            "ResNet34": (torchvision.models.resnet34, 512),
            "ResNet50": (torchvision.models.resnet50, 2048),
            "ResNet101": (torchvision.models.resnet101, 2048),
            "ResNet152": (torchvision.models.resnet152, 2048),
        }
        model_fn, in_features = resnet_models[model_arch]
        self.model = model_fn(weights=None).to(device)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features, num_classes)
        ).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        layer_map = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4"
        }

        for layer_name, enabled in layer_map.items():
            if enabled:
                layer = getattr(self.model, layer_map[layer_name])
                for param in layer.parameters():
                    param.requires_grad = True

        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(device)
        self.model = self.model.float()
        self.model.eval()

        return self.model
    
    def download_model(self, username, model_name):
        with open(USERS_JSON, 'r') as f:
            users = json.load(f)
        user_models = users[username].get("models", {})
        model_path = user_models[model_name]["path"]
        state_dict = torch.load(model_path, map_location=device)

        with open(f"{model_name}.pkl", "wb") as f:
            pickle.dump(state_dict, f)

        return
    
    def gradcam(self, username, image, model_name, num_classes):

        input_tensor = image.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        rgb_img = image.permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        rgb_img = rgb_img.astype("float32")

        loaded_model = self.load_model(username, model_name, num_classes)
        output = loaded_model(input_tensor)
        predicted_class = output.argmax().item()

        cam = GradCAM(model=loaded_model, target_layers=[loaded_model.layer4[-1]])
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb = True)
    
        return predicted_class, rgb_img, cam_image
    
    def gaussian_blur(self, img, sigma):
        k = int(2 * round(3 * sigma) + 1)
        x = torch.arange(k) - k // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel.expand(3, 1, k, k).to(img.device)
        return F.conv2d(img, kernel, padding=k//2, groups=3)
    
    def feature_visualization(
        self,
        layer_name: str,
        channel_idx: int,
        img_size: int=224,
        steps: int= 80,
        lr: float=0.1,
        tv_weight: float = 1e-4,
        l2_weight: float = 1e-4,
        device: str = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval().to(device)

        def get_layer(model, layer_name):
            modules = dict([*model.named_modules()])
            return modules.get(layer_name)

        target_layer = get_layer(self.model, layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model.")

        activation = None

        def hook_fn(module, input, output):
            nonlocal activation
            activation = output

        handle = target_layer.register_forward_hook(hook_fn)

        img = torch.randn(1, 3, img_size, img_size, device=device, requires_grad = True)
        optimizer = torch.optim.Adam([img], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()

            _ = self.model(img)

            if activation is None:
                continue

            loss = -activation[:, channel_idx].mean()

            loss += 1e-4 * torch.norm(img)

            tv = (
                torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
                torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
            )
            loss += 1e-4 * tv

            if step % 12 == 0 and step > 40:
                sigma = 0.5 + (step / steps) * 0.5
                img.data = self.gaussian_blur(img, sigma)

            loss.backward()
            optimizer.step()

            img.data = torch.clamp(img.data, -2.5, 2.5)
        
        handle.remove()

        img = img.detach().cpu().squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        return Image.fromarray(img)
    
    def get_activation_maps(self, model, layer_name, image_tensor):
        activations = {}

        def hook_fn(module, input, output):
            activations['feat'] = output.detach().cpu()
        
        layer = dict(model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook_fn)

        _ = model(image_tensor)

        handle.remove()

        return activations['feat']
    
    def activation_grid(self, activations, max_cols=8):
        acts = activations.squeeze(0)

        C, H, W = acts.shape

        acts = acts.clone()
        for i in range(C):
            ch = acts[i]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            acts[i] = ch
        
        acts = acts.cpu().numpy()
        cols = min(max_cols, C)
        rows = math.ceil(C / cols)
        grid = np.zeros((rows * H, cols * W), dtype=np.uint8)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= C:
                    break
                ch_img = (acts[idx] * 255).astype(np.uint8)
                grid[r*H:(r+1)*H, c*W:(c+1)*W] = ch_img
                idx += 1
        
        return Image.fromarray(grid)

    def get_cpu_ram_usage(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return cpu, ram
    
    def get_gpu_usage(self):
        try:
            result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv, noheader, nounits"]
            ).decode("utf-8").strip()

            gpu_util, mem_used, mem_total, temp = result.split(", ")

            return {
                "load": float(gpu_util),
                "memory_used": float(mem_used),
                "memory_total": float(mem_total),
                "temperature": float(temp)
            }
        except Exception:
            return None