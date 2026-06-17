import gradio as gr
from system.system_functions.modelhandler import ModelManager
from system.system_functions.securityhandler import SecurityManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
from torchvision import transforms
from system.backend_config.config import USER_DB
import json
import os
import pygame
from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, MUSIC_FOLDER

# Initialise sound system
pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/ping.mp3"
pygame.mixer.music.load(music_path)

# Instantiate managers
mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class Train_Tab():
    """UI + logic for training CNN models, visualizing metrics, and saving trained models."""
    def __init__(self, current_user):
            gr.Markdown("### Train Models.")

            # Available transform options for training
            transform_options = [
                "Resize(256, 256)",
                "CenterCrop(224)",
                "RandomResizedCrop(224)",
                "RandomHorizontalFlip",
                "RandomVerticalFlip",
                "RandomRotation",
                "ColorJitter",
                "RandomAffine",
                "ToTensor",
                "Normalize"
            ]

            # Available architectures
            architecture_options = [
                "ResNet18",
                "ResNet34",
                "ResNet50",
                "ResNet101",
                "ResNet152"
            ]

            self.current_user = current_user

            # Load user preferences
            prefs = pm.get_preferences(current_user)

            # Stop Flag
            self.stop_flag = gr.State(False)

            # States for saving final training results
            self.final_accuracy = gr.State()
            self.final_loss = gr.State()
            self.final_epochs = gr.State()

            # System analytics history
            self.gpu_history = []
            self.cpu_history = []
            self.ram_history = []

            # Dataset Input
            with gr.Group():
                gr.Markdown("Dataset Input")
                self.train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")

            # Hyperparameters
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate", value=prefs.get("default_learning_rate"))
                self.epoch_input = gr.Number(label="Epochs", value=prefs.get("default_epochs"))
                self.bs_input = gr.Number(label="Batch Size", value=prefs.get("default_batch_size"))
                self.dropout_input = gr.Slider(0.0, 0.7, step=0.05, label="Dropout Rate", value=prefs.get("default_dropout"))

            # Early Stopping
            with gr.Group():
                gr.Markdown("Early Stopping")
                self.earlystopping_input = gr.Checkbox(label="Enable Early Stopping", value=False)
                self.patience_input = gr.Slider(1, 20, step=1, label="Patience (Epochs)")
            
            # Transforms
            with gr.Group():
                gr.Markdown("Transforms")
                self.train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")

            # Architecture + Activation Layers
            with gr.Group():
                gr.Markdown("Architecture")
                self.archtype_input = gr.Dropdown(choices=architecture_options, label="Select preferred model architecture (The smaller the dataset, the smaller the architecture)", value=prefs.get("default_architecture"))
                with gr.Row():
                    
                    # Activation layer defaults
                    activation_defaults = prefs.get("default_activation_layer", "layer4")

                    if isinstance(activation_defaults, str):
                        activation_defaults = [activation_defaults]
                    self.layer1_input = gr.Checkbox(label="Use Layer1", value = "layer1" in activation_defaults)
                    self.layer2_input = gr.Checkbox(label="Use Layer2", value = "layer2" in activation_defaults)
                    self.layer3_input = gr.Checkbox(label="Use Layer3", value = "layer3" in activation_defaults)
                    self.layer4_input = gr.Checkbox(label="Use Layer4", value = "layer4" in activation_defaults)
            
            # Training Outputs
            with gr.Group():
                gr.Markdown("Training")
                self.train_btn = gr.Button("Start Training")
                self.stop_btn = gr.Button("Stop Training")
                with gr.Row(equal_height=True):
                    self.train_status = gr.Textbox(label="Status", lines=10)
                    self.train_graph = gr.Plot(label="Loss Curve")
                    self.acc_graph = gr.Plot(label="Accuracy Curve")
            
            # System Analytics
            with gr.Group():
                gr.Markdown("System Analytics")
                with gr.Row(equal_height = True):
                    self.gpu_plot = gr.Plot(label="GPU Usage (%)")
                    self.cpu_plot = gr.Plot(label = "CPU / RAM (%)")
                self.analytics_json = gr.JSON(label="Live Analytics")

            # Save Model
            with gr.Group():
                gr.Markdown("Save Model")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        self.save_btn = gr.Button("Save Model")
                        self.save_model_name = gr.Textbox(label="Saved Model Name", placeholder="model1")
                        self.save_status = gr.Textbox(label="Save Status", interactive=False)

            # Callbacks
            self.train_btn.click(
            fn=self.train_pipeline,
            inputs=[self.train_path_input, self.epoch_input, self.lr_input, self.bs_input, self.layer1_input, self.layer2_input, self.layer3_input, self.layer4_input, self.train_transforms_input, self.earlystopping_input, self.patience_input, self.archtype_input, self.dropout_input],
            outputs=[self.train_status, self.train_graph, self.acc_graph, self.gpu_plot, self.cpu_plot, self.analytics_json])

            self.stop_btn.click(
                fn=self.request_stop,
                inputs=[],
                outputs=[self.train_status]
            )

            self.save_btn.click(
            fn=self.save_model,
            inputs=[self.current_user, self.save_model_name, self.final_accuracy, self.final_loss, self.final_epochs],
            outputs=[self.save_status])
            
    def train_pipeline(self, train_folder, epochs, lr, bs, layer1, layer2, layer3, layer4, selected_transforms, earlystopping, patience, arch_type, dropout):
        """Full training pipeline, handling input validation, build transforms, load dataset, build model, train model, stream live metrics + analytics"""
        try:
            # Reset stop flag for new training session
            self.stop_flag.value = False
            # Handle Gradio File objects OR plain strings
            path = train_folder.name if hasattr(train_folder, "name") else train_folder

            # Validation
            if not path:
                return gr.Warning("Please enter a training folder path."), None, None, None, None, None
            
            if epochs is None or epochs <= 0:
                return gr.Warning("Epochs must be a positive integer."), None, None, None, None, None
            
            if lr is None or lr <= 0:
                return gr.Warning("Learning rate must be a positive number."), None, None, None, None, None
            
            if bs is None or bs <= 0:
                return gr.Warning("Batch size must be a positive number."), None, None, None, None, None
            
            if dropout is None or dropout < 0 or dropout > 1:
                return gr.Warning("Dropout must be between 0 and 1."), None, None, None, None, None
            
            if patience is None or patience <= 0:
                return gr.Warning("Patience must be a positive integer.")

            # Validate path security
            SecurityManager(path).validate_path()

            # Transforms
            if not selected_transforms or len(selected_transforms) == 0:
                train_transforms = None
            else:
                final_transforms = []

                # Base transforms
                final_transforms.append(transforms.Resize(256))
                final_transforms.append(transforms.CenterCrop(256))
                
                # Optional transforms
                if "RandomResizedCrop(224)" in selected_transforms:
                    final_transforms.append(transforms.RandomResizedCrop(224, 
                                                                        scale=(0.6, 1.0)))
                
                if "RandomHorizontalFlip" in selected_transforms:
                    final_transforms.append(transforms.RandomHorizontalFlip(0.5))

                if "RandomVerticalFlip" in selected_transforms:
                    final_transforms.append(transforms.RandomVerticalFlip(0.5))
                
                if "RandomRotation" in selected_transforms:
                    final_transforms.append(transforms.RandomRotation(360))

                if "ColorJitter" in selected_transforms:
                    final_transforms.append(transforms.ColorJitter(brightness=0.5, 
                                                                contrast=0.5, 
                                                                saturation=0.5, 
                                                                hue=0.1
                                                                ))
                if "RandomAffine" in selected_transforms:
                    final_transforms.append(transforms.RandomAffine(degrees=(-15, 15),     
                                                                    translate=(0.1, 0.1),  
                                                                    scale=(0.9, 1.1),      
                                                                    shear=(-5, 5),         
                                                                    fill=0                
                                                                    ))

                # Always convert to tensor + normalize
                final_transforms.append(transforms.ToTensor())
                final_transforms.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                )
                train_transforms = transforms.Compose(final_transforms)
            
            # Load Dataset + Build Model
            train_path = os.path.join(path, "train")
            mm.train_transforms_dataset(train_transforms, train_path, bs)
            mm.build(arch_type, layer1, layer2, layer3, layer4, dropout)

            # Start training generator
            gen =  mm.train(earlystopping, patience, epochs, lr, stop_callback = lambda: self.stop_flag.value)

            # Stream Training Output
            while True:
                if self.stop_flag.value:
                    yield "Training stopped by user.", None, None, None, None, None
                    break
                try:
                    log, losses, accuracies, analytics = next(gen)

                    if self.stop_flag.value:
                        yield "Training stopped by user.", None, None, None, None, None
                        break
                    
                    # Track system analytics
                    self.gpu_history.append(analytics["gpu"]["load"] if analytics["gpu"] else 0)
                    self.cpu_history.append(analytics["cpu"])
                    self.ram_history.append(analytics["ram"])

                    # Update plots
                    fig = gm.update_loss(losses, len(losses))
                    acc_fig = gm.update_accuracy(accuracies, len(accuracies))
                    gpu_fig = gm.update_gpu_plot(self.gpu_history)
                    cpu_ram_fig = gm.update_cpu_ram_plot(self.cpu_history, self.ram_history)

                    yield log, fig, acc_fig, gpu_fig, cpu_ram_fig, analytics

                except StopIteration as e:
                    # Final results
                    losses, accuracies = e.value
                    fig = gm.update_loss(losses, epochs)
                    acc_fig = gm.update_accuracy(accuracies, epochs)
                    gpu_fig = gm.update_gpu_plot(self.gpu_history)
                    cpu_ram_fig = gm.update_cpu_ram_plot(self.cpu_history, self.ram_history)

                    # Notifications
                    if NOTIFICATIONS_ENABLED:
                        gr.Info("Training completed successfully!", duration=8)
                    if SOUNDSENABLED:
                        pygame.mixer.music.play()

                    # Save final metrics
                    self.trained_model = mm.model
                    self.final_accuracy.value = accuracies[-1]
                    self.final_loss.value = losses[-1]
                    self.final_epochs.value = len(losses)

                    self.losses = losses
                    self.accuracies = accuracies
                    self.arch_type = arch_type

                    yield log, fig, acc_fig, gpu_fig, cpu_ram_fig, analytics
                    break

        # Error handling in case of Exception
        except Exception as e:
            yield gr.Warning(str(e)), None, None, None, None, None

    def save_model(self, user, model_name, accuracy, loss, epochs):
        """Save Trained Model"""
        try:
            # Check if trained model
            if not hasattr(self, "trained_model"):
                return gr.Warning("No model to train")
            
            # Check if model name passed
            if not model_name:
                return gr.Warning("Must pass model name")
            
            # Load DB
            with open(USER_DB, "r") as f:
                users = json.load(f)

            # Prevent overwriting existing model
            if model_name in users[user]["models"]:
                return gr.Warning(f"A model named '{model_name} already exists. Choose a different name.")
            
            # Notifications
            if NOTIFICATIONS_ENABLED:
                gr.Info("Saving completed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.play()

            # Save model + metadata
            return mm.save_model(self.trained_model, user, model_name, accuracy, loss, epochs, self.losses, self.accuracies, self.arch_type)
        
        # Error handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e))
    
    def request_stop(self):
        """Stops training if the user presses the stop button"""
        self.stop_flag.value = True
        if NOTIFICATIONS_ENABLED:
            gr.Info("Stopping training...")
        if SOUNDSENABLED:
            pygame.mixer.music.play()
    
    def refresh_preferences(self, user):
        """Refresh UI defaults based on user preferences."""
        prefs = pm.get_preferences(user)
        activation_defaults = prefs.get("default_activation_layer", "layer4")
        if isinstance(activation_defaults, str):
            activation_defaults = [activation_defaults]
        # Get defaults for LR, Epochs, Batch Size, Dropout, Architecture, Layers
        return (
            gr.update(value=prefs.get("default_learning_rate", 0.001)),
            gr.update(value=prefs.get("default_epochs", 10)),
            gr.update(value=prefs.get("default_batch_size", 32)),
            gr.update(value=prefs.get("default_dropout", 0.2)),
            gr.update(value=prefs.get("default_architecture", "ResNet18")),
            gr.update(value="layer1" in activation_defaults),
            gr.update(value="layer2" in activation_defaults),
            gr.update(value="layer3" in activation_defaults),
            gr.update(value="layer4" in activation_defaults),
        )
