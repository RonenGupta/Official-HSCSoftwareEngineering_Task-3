import gradio as gr
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
from system.system_functions.securityhandler import SecurityManager
import torch
import numpy as np
from torchvision import transforms
import os
import random
import json
import pygame
from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER

pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/ping.mp3"
pygame.mixer.music.load(music_path)

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class GradCAM():
    def __init__(self, current_user):
        gr.Markdown("### View GradCAMs.")
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
        self.current_user = current_user
        
        with gr.Group():
                gr.Markdown("Image Dataset")
                self.image_path_input = gr.Textbox(label="Image Folder Path", placeholder="/absolute/path/to/your/dataset")

        with gr.Group():
            gr.Markdown("Model Selection")
            self.model = gr.Dropdown(choices=[], label="Select a saved model for testing!", interactive=True)

        with gr.Group():
            gr.Markdown("Transforms")
            self.transforms = gr.CheckboxGroup(choices=transform_options, label="Select transforms for testing augmentations!")
        
        with gr.Group():
            gr.Markdown("Hyperparameters")
            self.bs_input = gr.Number(label="Batch Size")
        
        with gr.Group():
            gr.Markdown("Refresh Models")
            self.refresh_btn = gr.Button("Refresh Saved Models")

        with gr.Group():
            gr.Markdown("View Augmentations")
            self.augbutton = gr.Button("View Augmentation Examples")
            self.augmentation = gr.Image(type="numpy", buttons=["download"])
        
        with gr.Group():
            gr.Markdown("GradCAM Results")
            self.gcbutton = gr.Button("Generate GradCAM")
            with gr.Row(equal_height=True):
                self.originimage = gr.Image(type="numpy", label="Original Image", buttons=["download"], container=False)
                self.gradimage = gr.Image(type="numpy", label="GradCAM Image", buttons=["download"], container=False)
                self.predclass = gr.Label()

        self.refresh_btn.click(
            fn=self.get_user_models,
            inputs=[self.current_user],
            outputs=[self.model]
        )
        
        self.gcbutton.click(
            fn=self.update_gradcam,
            inputs=[self.current_user, self.image_path_input, self.model, self.bs_input],
            outputs=[self.predclass, self.originimage, self.gradimage]
        )

        self.augbutton.click(
            fn=self.update_augmentations,
            inputs=[self.image_path_input, self.transforms, self.bs_input],
            outputs=[self.augmentation]
        )

    def update_gradcam(self, username, dataset_path, model_name, bs):
        try:

            if not model_name:
                gr.Warning("Please select a model before generating GradCAM.")
                return None, None, None
            
            if not dataset_path:
                gr.Warning("Please enter a dataset path.")
                return None, None, None
            
            SecurityManager(dataset_path).validate_path()

            if bs is None or bs <= 0:
                gr.Warning("Batch size must be a positive number.")
                return None, None, None
            
            gradcam_path = os.path.join(dataset_path, "test")
            class_names = mm.test_transforms_dataset(None, gradcam_path, bs)
            num_classes = len(class_names)

            if not hasattr(mm, 'test_dataset') or mm.test_dataset is None:
                return gr.Warning("Failed to load test dataset. Check folder structure.")
            
            if len(mm.test_dataset) == 0:
                return gr.Warning("Test dataset is empty."), None, None
            
            index = random.randint(0, len(mm.test_dataset) - 1) 
            pil_image, _ = mm.test_dataset[index]

            predicted_class, originimage, cam_image = mm.gradcam(username, pil_image, model_name, num_classes)
            predicted_label = class_names[predicted_class].capitalize()

            if NOTIFICATIONS_ENABLED:
                gr.Info("GradCAM computation completed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.play()
            return predicted_label, originimage, cam_image
        
        except Exception as e:
            return gr.Warning(str(e)), None, None
        
    def update_augmentations(self, augmentation_path, selected_transforms, bs):
        try:
            if not selected_transforms:
                return gr.Warning("No augmentation transforms supplied.")
            
            aug_transforms = []

            if "Resize(256, 256)" in selected_transforms:
                aug_transforms.append(transforms.Resize((224, 224)))
            
            if "CenterCrop(224)" in selected_transforms:
                aug_transforms.append(transforms.CenterCrop(224))
            
            if "RandomResizedCrop(224)" in selected_transforms:
                aug_transforms.append(transforms.RandomResizedCrop(224, scale=(0.6, 1.0)))
            
            if "RandomHorizontalFlip" in selected_transforms:
                aug_transforms.append(transforms.RandomHorizontalFlip(0.5))

            if "RandomVerticalFlip" in selected_transforms:
                aug_transforms.append(transforms.RandomVerticalFlip(0.5))
            
            if "RandomRotation" in selected_transforms:
                aug_transforms.append(transforms.RandomRotation(360))
            
            if "ColorJitter" in selected_transforms:
                aug_transforms.append(transforms.ColorJitter(
                                                            brightness=0.5, 
                                                            contrast=0.5, 
                                                            saturation=0.5, 
                                                            hue=0.1
                                                            ))
            
            if "RandomAffine" in selected_transforms:
                aug_transforms.append(transforms.RandomAffine(degrees=(-15, 15),     
                                                                    translate=(0.1, 0.1),  
                                                                    scale=(0.9, 1.1),      
                                                                    shear=(-5, 5),         
                                                                    fill=0                
                                                                    ))
            if not any(isinstance(t, transforms.ToTensor) for t in aug_transforms):
                aug_transforms.append(transforms.ToTensor())
            augmented_transforms = transforms.Compose(aug_transforms)
            augmentation_path = os.path.join(augmentation_path, "test")
            mm.test_transforms_dataset(augmented_transforms, augmentation_path, bs)

            if not hasattr(mm, 'test_dataset') or mm.test_dataset is None:
                return gr.Warning("Failed to load dataset for augmentation.")
            dataset = mm.test_dataset
            index = random.randint(0, len(dataset) - 1) 
            pil_image, _ = dataset[index]

            if isinstance(pil_image, torch.Tensor):
                if pil_image.dim() == 3:
                    pil_image = pil_image.permute(1, 2, 0).numpy()
                else:
                    pil_image = pil_image.numpy()
            elif hasattr(pil_image, 'convert'):
                pil_image = np.array(pil_image)
                
            if NOTIFICATIONS_ENABLED:
                gr.Info("Augmentated image computed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.play()

            return pil_image
        except Exception as e:
            return gr.Warning(str(e))
    
    def get_user_models(self, username):
        try:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            model_names = list(users[username]["models"].keys())
            return gr.update(choices=model_names, value=None)
        except Exception as e:
            return gr.Warning(str(e))