import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import json
import math
import time
import pickle
import datetime
import psutil
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Set a random seed for pytorch to ensure results in training are reproducible
torch.manual_seed(42)

# Set the best available hardware for pytorch
if torch.cuda.is_available():
    device = torch.device("cuda") # Uses GPU if available (fastest option)
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Uses metal performance shaders (strictly for Macbook M series) if available
else:
    device = torch.device("cpu") # Uses CPU if available (slowest option)

USERS_JSON = Path("users.json") # Get path for user data

class ModelManager():
    """Handles all operations
    according to CNN's, such as model training, 
    dataset preparation, model explainability tools,
    model downloading, saving, deleting"""
    def __init__(self):
        # No initialisation needed, but class structure kept for clarity
        pass

    def test(self, model):
        """Testing loop, takes a model object,
        returns test loss, metrics, preds and labels for plots"""
        # Check if model exists
        if model is None:
            raise ValueError("No model loaded for testing.")
        
        # Check if test dataloader exists
        if not hasattr(self, "test_dataloader"):
            raise RuntimeError("Test dataloader not initialized.")
        
        try:
            # Initialise a loss function, test loss and accuracy, preds and labels list
            loss_fn = torch.nn.CrossEntropyLoss()
            test_loss = 0
            test_acc = 0
            all_preds = []
            all_labels = []
            # Disable gradient tracking, as we are doing testing for inference only
            with torch.no_grad():
                # Loop through the test dataloader (only once as it is inference)
                for X, y in self.test_dataloader:
                    # Move input tensors to the same device and prevent device mismatching
                    X = X.to(device)
                    y = y.to(device)

                    # Run a forward pass to compute raw prediction scores
                    y_pred = model(X)
                    # Convert raw scores into predicted class labels
                    _, preds = torch.max(y_pred, 1)

                    # Move all model predicted class labels and true labels to CPU (as numpy only works on CPU),
                    # convert to numpy array, and add them to lists called all_preds and all_labels
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

                    # Compute loss from model prediction and true label, previously
                    # created loss function
                    loss = loss_fn(y_pred, y)

                    # Add loss to the total test loss
                    test_loss += loss.item()
                    # Add accuracy to the total accuracy by converting to numpy and computing score from
                    # sklearn method
                    test_acc += accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())

                # Compute average test loss and accuracy
                avg_test_loss = test_loss / len(self.test_dataloader)
                avg_test_acc = test_acc / len(self.test_dataloader)
                # Compute test metrics (Loss, Accuracy, Precision, Recall, F1 Score)
                test_metrics = f"Test Loss: {avg_test_loss} || Test Accuracy: {avg_test_acc}\n Test Precision: {precision_score(all_labels, all_preds, average='macro')}\nTest Recall: {recall_score(all_labels, all_preds, average='macro')}\nTest F1-Score: {f1_score(all_labels, all_preds, average='macro')}"
            
            # Return test metrics, true labels, prediction labes
            return test_metrics, all_labels, all_preds
        # Error handling if testing failed due to a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Testing failed: {str(e)}")

    def train(self, earlystopping, patience, epochs: int, lr: float = 0.01):
        """Training loop, takes user inputted early stopping boolean,
        patience if supplied, epochs, learning rate
        returns training log, losses, accuracies, analytics"""
        # Check if train dataloader exists
        if not hasattr(self, "train_dataloader"):
            raise RuntimeError("Training dataloader not initialised.")
        # Check if model exists
        if self.model is None:
            raise RuntimeError("Model not built before training.")
        
        # Initialise log, param_list
        log = ""
        param_list = []

        # Loops through all model param tensors and checks if they are trainable
        # if so, appends them to the param_list
        for param in self.model.parameters():
            if param.requires_grad:
                param_list.append(param)
        try:
            # Initialise a loss function
            loss_fn = torch.nn.CrossEntropyLoss()
            # Optimizer reads gradients from backprop, updates model weights, uses lr and paramlist 
            # to determine speed of training and what it is allowed to update
            optimizer = torch.optim.Adam(param_list, lr, weight_decay=1e-4)

            # Initialise a best loss placeholder and early stopping counter
            best_loss = float('inf')
            early_stop_counter = 0

            # Set epochs, losses and accuracies
            EPOCHS = epochs
            losses = []
            accuracies= []

            # Loop through each epoch
            for epoch in range(EPOCHS):
                # Initialise current train loss and accuracy zero
                train_loss = 0
                train_acc = 0

                # Put model in training mode, enabling gradient tracking
                self.model.train()
                # Loop through train dataloader
                for X, y in self.train_dataloader:
                    
                    # Marks the start of one training iteration / batch in the dataloader
                    step_start = time.time()

                    # Move input tensors to the same device and prevent device mismatching
                    X = X.to(device)
                    y = y.to(device)

                    # Run a forward pass to compute raw prediction scores
                    y_pred = self.model(X)

                    # Compute loss from model prediction and true label, previously
                    # created loss function
                    loss = loss_fn(y_pred, y)

                    # Add loss to the total train loss
                    train_loss += loss.item()
                    # Add accuracy to the total accuracy by converting to numpy and computing score from
                    # sklearn method
                    train_acc += accuracy_score(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())

                    # Resets all gradients in the model to zero to ensure gradient stacking does not occur
                    optimizer.zero_grad()

                    # Perform backpropogation on weights to reduce loss
                    loss.backward()

                    # Update model weights using the gradients computed above
                    optimizer.step()

                    # Computes the time taken to go through one iteration / batch
                    step_time = time.time() - step_start
                    # Converts the batch duration into iterations per second
                    iters_per_sec = 1 / step_time

                    # Get CPU/RAM and GPU usage from inclass methods
                    cpu, ram = self.get_cpu_ram_usage()
                    gpu = self.get_gpu_usage()

                    # Make analytics skeleton with calculated data
                    analytics = {
                        "epoch": epoch + 1,
                        "step_time": step_time,
                        "iters_per_sec": iters_per_sec,
                        "cpu": cpu,
                        "ram": ram,
                        "gpu": gpu
                    }
                
                # Compute average train loss and accuracy
                avg_train_loss = train_loss / len(self.train_dataloader)
                avg_train_acc = train_acc / len(self.train_dataloader)
                
                # Add train loss and accuracy to predefined lists for plotting
                losses.append(avg_train_loss)
                accuracies.append(avg_train_acc)

                # Create log with all metrics
                log += f"Epoch: {epoch + 1} || Train Loss: {avg_train_loss} || Train Accuracy {avg_train_acc}\n"

                # Check if early stopping enabled
                if earlystopping:
                    # Checks whether model improved this epoch
                    # If so, adds message to log and updates best loss and reset patience counter
                    if avg_train_loss < best_loss:
                        best_loss = avg_train_loss
                        early_stop_counter = 0
                        log += "Best loss improved"
                    else:
                        # If model did not improve, add one to patience counter
                        # and compute percentage for patience
                        early_stop_counter += 1
                        log += f"No improvement ({early_stop_counter/patience})"
                        
                        # If the patience counter exceeds the original patience given
                        # stop training
                        if early_stop_counter >= patience:
                            log += f"\nEarly stopping triggered at epoch {epoch+1}!"
                            break
                
                # Adds a new line for readability
                log += "\n"

                # Yields metrics each epoch rather than after all epochs are finished
                # streaming live updates to Gradio UI, updating in real time
                yield log, losses, accuracies, analytics
            # After training finishes, return final results of training
            return losses, accuracies
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Training failed {str(e)}")

    def build(self, architecture: str = "ResNet18", layer1: bool = False, layer2: bool = False, layer3: bool = False, layer4: bool = False, dropout=0.2):
        """Model build process, which takes in preferred architecture and dropout,
        builds a ResNet model, unfreezes chosen layers and stores configuration"""
        # Check if training dataset exists
        if not hasattr(self, "train_dataset"):
            raise RuntimeError("Training dataset not loaded before buildinng model.")
        try:
            # Dictionary of resnet models with constructor function and number of output features required for fc layer
            resnet_models = {
                "ResNet18": (torchvision.models.resnet18, 512),
                "ResNet34": (torchvision.models.resnet34, 512),
                "ResNet50": (torchvision.models.resnet50, 2048),
                "ResNet101": (torchvision.models.resnet101, 2048),
                "ResNet152": (torchvision.models.resnet152, 2048),
            }
            # Picks correct ResNet and loads pretrained weights, moves model to chosen device
            model_fn, in_features = resnet_models[architecture]
            self.model = model_fn(weights="DEFAULT").to(device)
            # Replaces final fc layer with chosen dropout, in_features, out_features
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout, inplace=True),
                torch.nn.Linear(in_features=in_features,
                        out_features=len(self.train_dataset.classes),
                        bias=True)).to(device)
            
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze only the final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True

            # Map booleans to layer name
            layer_map = {
                layer1: "layer1",
                layer2: "layer2",
                layer3: "layer3",
                layer4: "layer4"
            }

            # Unfreeze selected layers
            for name, key in layer_map.items():
                if name:
                    layer = getattr(self.model, key)
                    for param in layer.parameters():
                        param.requires_grad = True
            
            # Save the configuration
            self.layer_config = {
                "layer1": layer1,  
                "layer2": layer2,
                "layer3": layer3,
                "layer4": layer4     
            }
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Model build failed: {str(e)}")
    
    def test_transforms_dataset(self, test_transforms: transforms.Compose | None, test_path: str, test_bs: int = 32):
        """Setup dataloader for testing data, takes transforms, testing dataset path, testing batch size"""
        # Check if testing batch size is less than or equal to zero
        if test_bs <= 0:
            raise ValueError("Batch size must be greater than zero.")
        try:
            # If not testing transforms were supplied, use defaults
            if test_transforms is None:
                test_transforms = transforms.Compose([
                                                    transforms.Resize((224, 224)),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                        std=[0.229, 0.224, 0.225])
                                                    ])
            # Loading testing images from path, assign labels based on folder names, apply testing transforms 
            self.test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform = test_transforms)
            # Wraps dataset and handles batching of the dataset from user input
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=test_bs, shuffle=False)
            # Return test_dataset classes for confusion matrix plotting, data store
            return self.test_dataset.classes
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to load test dataset: {e}")
    
    def train_transforms_dataset(self, train_transforms: transforms.Compose | None, train_path: str, train_bs: int = 32,):
        """Setup dataloader for training data, takes transforms, training dataset path, training batch size"""
        # Check if training batch size is less than or equal to zero
        if train_bs <= 0:
            raise ValueError("Batch size must be greater than zero.")
        try:
            # If no training transforms were supplied, use defaults
            if train_transforms is None:
                train_transforms = transforms.Compose([
                                                    transforms.Resize(256),                    
                                                    transforms.CenterCrop(256),
                                                    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                                                    transforms.RandomHorizontalFlip(0.5),
                                                    transforms.ColorJitter(
                                                        brightness=0.4,
                                                        contrast=0.4,
                                                        saturation=0.4,
                                                        hue=0.1
                                                    ),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                        std=[0.229, 0.224, 0.225])
                                                    ])
            # Loading training images from path, assign labels based on folder names, apply training transforms 
            self.train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform = train_transforms)
            # Wraps dataset and handles batching of the dataset from user input, shuffling each epoch
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True)
        
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to load training dataset {e}")

    def save_model(self, model, username, model_name, final_accuracy, final_loss, epochs, loss_curve, accuracy_curve, arch_type):
        """Saves a trained model and its metadata and updates users.json"""
        try:
            # Load the users.json file into memory
            with open(USERS_JSON, "r") as f:
                users = json.load(f)

            # Check if the username exists in the database
            if username not in users:
                return "User not found"
            
            # Get the user's saved models dictionary (or empty dict if none exist)
            models = users[username].get("models", {})

            # If the model name already exists, reuse its saved file path
            if model_name in models:
                model_path = Path(models[model_name]["path"])
            else:
                # Otherwise, create a new file path for saving the model weights
                model_path = Path(f"saved_models/{username}_{model_name}.pth")
                # Create metadata entry for this new model
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

                # Add a default notes field for the user to edit later
                models[model_name]["notes"] = "My honest reaction:"
                # Save updated model metadata back into the user's entry
                users[username]["models"] = models

                # Write the updated users.json file to disk
                with open (USERS_JSON, "w") as f:
                    json.dump(users, f, indent=4)

            # Save the actual PyTorch model
            torch.save(model.state_dict(), model_path)

            return f"Model '{model_name}' saved!"
        
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
        
    def load_model(self, username, model_name, num_classes = 2):
        """Load a previously saved model for a given user. Takes username, model name, num classes
        returns a reconstructed model with loaded weights"""
        try:
            # Load users.json to access saved model metadata
            with open(USERS_JSON, 'r') as f:
                users = json.load(f)
            
            # Retrieve this user's saved models
            user_models = users[username].get("models", {})

            # Extract model path, architecture, and layer configuration
            model_path = user_models[model_name]["path"]
            model_arch = user_models[model_name]["architecture"]
            layer_config = user_models[model_name]["layers"]

            # Map architecture name to constructor + feature size
            resnet_models = {
                "ResNet18": (torchvision.models.resnet18, 512),
                "ResNet34": (torchvision.models.resnet34, 512),
                "ResNet50": (torchvision.models.resnet50, 2048),
                "ResNet101": (torchvision.models.resnet101, 2048),
                "ResNet152": (torchvision.models.resnet152, 2048),
            }

            # Build the correct ResNet architecture without pretrained weights
            model_fn, in_features = resnet_models[model_arch]
            self.model = model_fn(weights=None).to(device)

            # Rebuild the final fc layer to match the saved model
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features, num_classes)
            ).to(device)

            # Freeze all layers by default
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze only the layers that were originally unfrozen during training
            for layer_name, enabled in layer_config.items():
                if enabled:
                    layer = getattr(self.model, layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True

            # Load saved weights into the model
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

            # Ensure model is on correct device and in eval mode
            self.model = self.model.to(device)
            self.model = self.model.float()
            self.model.eval()

            # Return model
            return self.model
        
        # Error handling in case of RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def download_model(self, username, model_name):
        """Export a saved model's state_dict as a downloadable
        .pkl file. Takes username and model name, returns the output file path
         of the .pkl file"""
        try:
            # Load users.json to find the model path
            with open(USERS_JSON, 'r') as f:
                users = json.load(f)
            
            # Validate user exists
            if username not in users:
                raise ValueError("User not found.")
            
            # Retrieve user's saved models
            user_models = users[username].get("models", {})

            # Validate model exists
            if model_name not in user_models:
                raise ValueError("Model not found for this user.")
            
            # Get the saved .pth file path
            model_path = user_models[model_name]["path"]

            # Ensure the file actually exists
            if not Path(model_path).exists():
                raise FileNotFoundError("Saved model file does not exist.")
            
            # Load the model weights
            state_dict = torch.load(model_path, map_location=device)

            # Save them as a .pkl file for download
            out_path = f"{model_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(state_dict, f)

            # Return the .pkl file path
            return out_path
        
        # Error handling for RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    def gradcam(self, username, image, model_name, num_classes):
        """Generate Grad-CAM heatmap for a given image and saved model.
        Takes a username, input image, model name, and number of output classes
        for the model, returns predicted class, normal rgb image, and Grad-CAM heatmap overlay."""
        try:

            # Add batch dimension and move to device
            input_tensor = image.unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            # Convert tensor to normalized RGB image for overlay
            rgb_img = image.permute(1, 2, 0).cpu().numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            rgb_img = rgb_img.astype("float32")

            # Load the trained model
            loaded_model = self.load_model(username, model_name, num_classes)

            # Forward pass
            output = loaded_model(input_tensor)
            predicted_class = output.argmax().item()
            
            # Set up GradCAM on the last block of layer4
            cam = GradCAM(model=loaded_model, target_layers=[loaded_model.layer4[-1]])

            # Target the predicted class
            targets = [ClassifierOutputTarget(predicted_class)]

            # Generate GradCAM heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # Overlay heatmap on original image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb = True)
            
            # Return predicted class, original image, GradCAM image
            return predicted_class, rgb_img, cam_image
        
        # Error handling in case of RuntimeError
        except Exception as e:
            raise RuntimeError(f"GradCAM failed: {e}")
        
    def gaussian_blur(self, img, sigma):
        """Apply a Gaussian blur to an image tensor using a manually constructed kernel.
        Takes in an input image of type tensor, and the standard deviation of the Gaussian kernel,
        determining how blurry it would be and returns the blurred image tensor."""
        # Compute kernel size based on sigma
        k = int(2 * round(3 * sigma) + 1)
        # 1D Gaussian kernel
        x = torch.arange(k) - k // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()

        # Convert to 2D kernel
        kernel = gauss[:, None] * gauss[None, :]

        # Expand to 3 channels
        kernel = kernel.expand(3, 1, k, k).to(img.device)

        # Return blurred image tensor
        return F.conv2d(img, kernel, padding=k//2, groups=3)
    
    def feature_visualization(
        self,
        layer_name: str,
        channel_idx: int,
        img_size: int=224,
        steps: int= 80,
        lr: float=0.1,
        device: str = None
    ):
        """Generate an image that maximally activates a specific CNN channel.
        Takes in layer name, channel index, image size, learning rate, and device, and returns a
        normalized RGB image representing the feature visualization.
        """
        try:
            # Pick device if not provided
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Ensure model is in eval mode
            self.model.eval().to(device)

            # Helper to fetch a layer by name
            def get_layer(model, layer_name):
                modules = dict([*model.named_modules()])
                return modules.get(layer_name)

            # Get the target layer
            target_layer = get_layer(self.model, layer_name)
            if target_layer is None:
                raise ValueError(f"Layer {layer_name} not found in model.")

            activation = None
            
            # Hook to capture layer output
            def hook_fn(module, input, output):
                nonlocal activation
                activation = output

            # Register hook
            handle = target_layer.register_forward_hook(hook_fn)

            # Start with random noise image
            img = torch.randn(1, 3, img_size, img_size, device=device, requires_grad = True)

            # Optimize the image itself
            optimizer = torch.optim.Adam([img], lr=lr)

            for step in range(steps):
                optimizer.zero_grad()

                # Forward pass
                _ = self.model(img)

                # If hook hasn't fired yet, skip
                if activation is None:
                    continue
                
                # Maximize activation of chosen channel
                loss = -activation[:, channel_idx].mean()

                # L2 regularization (keeps image from exploding)
                loss += 1e-4 * torch.norm(img)

                # Total variation regularization (smooths noise)
                tv = (
                    torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
                    torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
                )
                loss += 1e-4 * tv

                # Optional blur later in optimization to smooth artifacts
                if step % 12 == 0 and step > 40:
                    sigma = 0.5 + (step / steps) * 0.5
                    img.data = self.gaussian_blur(img, sigma)

                # Backprop + update
                loss.backward()
                optimizer.step()

                # Clamp to keep values reasonable
                img.data = torch.clamp(img.data, -2.5, 2.5)
            
            # Remove hook
            handle.remove()

            # Normalize image to 0-255 for display
            img = img.detach().cpu().squeeze()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # Return normalized RGB image
            return Image.fromarray(img)
        
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Feature visualization failed: {e}")
    def get_activation_maps(self, model, layer_name, image_tensor):
        """Extract activation maps from a specific layer during a forward pass. Takes model, 
        layer_name, image of type tensor, returns activation maps"""
        try:
            # Convert model layers to dictionary
            modules = dict(model.named_modules())
            # Ensure layer exists
            if layer_name not in modules:
                raise ValueError(f"Layer '{layer_name}' not found")
            
            activations = {}

            # Hook to capture output
            def hook_fn(module, input, output):
                activations['feat'] = output.detach().cpu()

            # Register hook
            handle = modules[layer_name].register_forward_hook(hook_fn)

            # Forward pass
            _ = model(image_tensor)

            # Remove hook
            handle.remove()

            # Return activation maps
            return activations['feat']
        
        # Error handling in case of RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to extract activation maps: {e}")
    
    def activation_grid(self, activations, max_cols=8):
        """Convert activation maps into a grid image for visualization. Takes the calculated activations,
        maximum number of cols in the output grid, and returns the grid image."""
        try:
            # Remove batch dimension to (Channel, Height, Width)
            acts = activations.squeeze(0)

            C, H, W = acts.shape

            # Normalize each channel to 0-1
            acts = acts.clone()
            for i in range(C):
                ch = acts[i]
                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                acts[i] = ch
            
            # Convert to numpy
            acts = acts.cpu().numpy()

            # Determine grid size
            cols = min(max_cols, C)
            rows = math.ceil(C / cols)

            # Create empty grid
            grid = np.zeros((rows * H, cols * W), dtype=np.uint8)

            # Fill grid with channel images
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= C:
                        break
                    ch_img = (acts[idx] * 255).astype(np.uint8)
                    grid[r*H:(r+1)*H, c*W:(c+1)*W] = ch_img
                    idx += 1

            # Return grid image
            return Image.fromarray(grid)
        
        # Error handling in case of RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to build activation grid: {e}")
        
    def get_cpu_ram_usage(self):
        """Return current CPU and RAM usage percentages."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return cpu, ram
    
    def get_gpu_usage(self):
        """Return GPU load, memory usage, and temperature using nvidia-smi.
        Returns None if GPU is unavailable."""
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
        
    def layer_exists(self, model, layer_name):
        """Check whether a nested layer exists, supporting
        dot notations like 'layer4.0.conv1'"""
        try:
            parts = layer_name.split(".")
            module = model
            for p in parts:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)
            return True
        except Exception:
            return False