import gradio as gr
from modelhandler import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager
from torchvision import transforms
import threading
import time
import datetime
import bcrypt
import os
import json
import random
import pygame

USER_DB = "users.json"

pygame.mixer.init()
music_path = "/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/music/LevinIntro.mp3"
pygame.mixer.music.load(music_path)

NOTIFICATIONS_ENABLED = True
SOUNDSENABLED = True
CURRENTVOLUME = 0.5

mm = ModelManager()
gm= GraphManager()

class Dashboard():
    def __init__(self, current_user):
        self.current_user = current_user

        gr.Markdown(
            """
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1 style ='margin-bottom: 0; font-size: 6rem; font-weight: 700; color: #1d1d1f; letter-spacing: -0.02em'> MyCNN Dashboard </h1>
                <p style='font-size: 1.1rem; color: #555;'>Your central hub for training, testing, and managing convolutional models.</p>
            </div>
            """
        )

        with gr.Group():
            self.welcome = gr.Markdown(
                                    """
                                    <div style='padding: 15px; border-radius: 10px; background: #ffffff; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);'>
                                        <h3>Welcome!</h3>
                                        <p>Loading user data...</p>
                                    </div>
                                    """
                                )
        
        with gr.Row():
            with gr.Column():
                self.model_count = gr.Number(label="Saved Models", interactive=False)
            with gr.Column():
                self.last_model = gr.Textbox(label = "Last Trained Model", interactive=False)
            with gr.Column():
                self.last_accuracy = gr.Textbox(label="Last Accuracy", interactive=False)
            with gr.Column():
                self.last_time = gr.Textbox(label="Last Time Trained", interactive=False)
        
        with gr.Group():
            gr.Markdown("### Your Models")
            with gr.Column() as self.model_cards:  
                self.card_slots = []
                cards_per_row = 2
                for i in range(0, 10, cards_per_row):
                    with gr.Row(equal_height=True):
                        for j in range(cards_per_row):
                            if i+j >= 10:
                                break
                            
                            with gr.Column():
                                with gr.Accordion(label=f"Model {i+j+1}", open=False,visible=False) as acc:
                                    slot_md = gr.Textbox(value="", visible=False, interactive=False, lines=0, show_label=False)
                                    slot_loss = gr.Plot(visible=False)
                                    slot_acc = gr.Plot(visible=False)
                                    slot_cm = gr.Plot(visible=False)
                                    self.card_slots.append((acc, slot_md, slot_loss, slot_acc, slot_cm))

    def load_dashboard(self, username):
        if not username:
            return "<h3>Not logged in</h3>", 0, "-", "-"
        
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        models = users[username]["models"]

        count = len(models)

        if count == 0:
            return (
                f"<h3>Welcome {username}!</h3><p>Noooooo noooooo train a model!! :p</p>",
                0,
                "No models yet :(",
                "N/A",
                "N/A",
                {}
            )
        
        last_model = list(models.keys())[-1]
        last_acc = models[last_model].get("accuracy", "Unknown")
        last_time = models[last_model].get("date", "Unknown")

        return (
            f"<h3 style='font-size: 2rem;'>Welcome {username}!</h3><p>Here's your latest model activity.</p>",
            count,
            last_model,
            last_acc,
            last_time,
            models
        )
    
    def build_model_cards(self, models):
        
        updates = []

        for _ in self.card_slots:
            updates.extend([
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ])
        
        for i, (model_name, data) in enumerate(models.items()):
            if i >= len(self.card_slots):
                break

            html = (
            f"Model: {model_name}\n"
            f"Accuracy: {data.get('accuracy', 'Unknown')}\n"
            f"Loss: {data.get('loss', 'Unknown')}\n"
            f"Epochs: {data.get('epochs', 'Unknown')}\n"
            f"Date: {data.get('date', 'Unknown')}\n"
        )

            base = i * 5

            updates[base] = gr.update(visible=True)

            updates[base+1] = gr.update(value=html, visible=True)
            

            if data.get("loss_curve"):
                loss_fig = gm.update_loss(data["loss_curve"], len(data["loss_curve"]))
                updates[base + 2] = gr.update(value=loss_fig, visible=True)

            if data.get("accuracy_curve"):
                acc_fig = gm.update_accuracy(data["accuracy_curve"], len(data["accuracy_curve"]))
                updates[base + 3] = gr.update(value=acc_fig, visible=True)

            if data.get("confusion_matrix"):
                labels, preds = data["confusion_matrix"]
                class_names = data.get("class_names", [])
                cm_fig = gm.update_confusion_matrix(labels, preds, class_names)
                updates[base + 4] = gr.update(value=cm_fig, visible=True)

        return updates
    
    def get_card_components(self):
        return [c for slot in self.card_slots for c in slot]
    
class Train_Tab():
    def __init__(self, current_user):
            gr.Markdown("### Train Models.")
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

            architecture_options = [
                "ResNet18",
                "ResNet34",
                "ResNet50",
                "ResNet101",
                "ResNet152"
            ]

            self.current_user = current_user
            self.final_accuracy = gr.State()
            self.final_loss = gr.State()
            self.final_epochs = gr.State()

            with gr.Group():
                gr.Markdown("Dataset Input")
                self.train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate")
                self.epoch_input = gr.Number(label="Epochs")
                self.bs_input = gr.Number(label="Batch Size")
            with gr.Group():
                gr.Markdown("Early Stopping")
                self.earlystopping_input = gr.Checkbox(label="Enable Early Stopping", value=False)
                self.patience_input = gr.Slider(0, 20, step=1, label="Patience (Epochs)")
            with gr.Group():
                gr.Markdown("Transforms")
                self.train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")
            with gr.Group():
                gr.Markdown("Architecture")
                self.archtype_input = gr.Dropdown(choices=architecture_options, label="Select preferred model architecture (The smaller the dataset, the smaller the architecture)")
                with gr.Row():
                    self.layer1_input = gr.Checkbox(label="Use Layer1")
                    self.layer2_input = gr.Checkbox(label="Use Layer2")
                    self.layer3_input = gr.Checkbox(label="Use Layer3")
                    self.layer4_input = gr.Checkbox(label="Use Layer4")
            
            with gr.Group():
                gr.Markdown("Training")
                self.train_btn = gr.Button("Start Training")
                with gr.Row(equal_height=True):
                    self.train_status = gr.Textbox(label="Status", lines=10)
                    self.train_graph = gr.Plot(label="Loss Curve")
                    self.acc_graph = gr.Plot(label="Accuracy Curve")
            with gr.Group():
                gr.Markdown("Save Model")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        self.save_btn = gr.Button("Save Model")
                        self.save_model_name = gr.Textbox(label="Saved Model Name", placeholder="model1")
                        self.save_status = gr.Textbox(label="Save Status", interactive=False)

            self.train_btn.click(
            fn=self.train_pipeline,
            inputs=[self.train_path_input, self.epoch_input, self.lr_input, self.bs_input, self.layer1_input, self.layer2_input, self.layer3_input, self.layer4_input, self.train_transforms_input, self.earlystopping_input, self.patience_input, self.archtype_input],
            outputs=[self.train_status, self.train_graph, self.acc_graph])

            self.save_btn.click(
            fn=self.save_model,
            inputs=[self.current_user, self.save_model_name, self.final_accuracy, self.final_loss, self.final_epochs],
            outputs=[self.save_status])
            
    def train_pipeline(self, train_folder, epochs, lr, bs, layer1, layer2, layer3, layer4, selected_transforms, earlystopping, patience, arch_type):
        
        self.arch_type = arch_type
        path = train_folder.name if hasattr(train_folder, "name") else train_folder

        final_transforms = []

        if "Resize(256, 256)" in selected_transforms:
            final_transforms.append(transforms.Resize((224, 224)))
        
        if "CenterCrop(224)" in selected_transforms:
            final_transforms.append(transforms.CenterCrop(224))
        
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

        if "ToTensor" in selected_transforms:
            final_transforms.append(transforms.ToTensor())
        
        if "Normalize" in selected_transforms:
            final_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))
            
        if not final_transforms:
            train_transforms = None
        else:
            train_transforms = transforms.Compose(final_transforms)

        if not SecurityManager(path).validate_path():
            return "Invalid training folder", None
        
        train_path = os.path.join(path, "train")
        mm.train_transforms_dataset(train_transforms, train_path, bs)
        mm.build(arch_type, layer1, layer2, layer3, layer4)

        gen =  mm.train(earlystopping, patience, epochs, lr)
        try:
            while True:
                log, losses, accuracies = next(gen)
                fig = gm.update_loss(losses, len(losses))
                acc_fig = gm.update_accuracy(accuracies, len(accuracies))
                yield log, fig, acc_fig
        except StopIteration as e:
            losses, accuracies = e.value
            fig = gm.update_loss(losses, epochs)
            acc_fig = gm.update_accuracy(accuracies, epochs)
            yield log, fig, acc_fig

            self.losses = losses
            self.accuracies = accuracies

            if NOTIFICATIONS_ENABLED:
                gr.Info("Training completed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.play()

        self.trained_model = mm.model
        self.final_accuracy.value = accuracies[-1]
        self.final_loss.value = losses[-1]
        self.final_epochs.value = len(losses)

    def save_model(self, user, model_name, accuracy, loss, epochs):
        if not hasattr(self, "trained_model"):
            return "No model to train"
        
        if not model_name:
            return "Must pass model name"
        if NOTIFICATIONS_ENABLED:
            gr.Info("Saving completed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()
        return mm.save_model(self.trained_model, user, model_name, accuracy, loss, epochs, self.losses, self.accuracies, self.arch_type)

class Test_Tab():
    def __init__(self, current_user):
            gr.Markdown("### Test Models.")
            self.current_user = current_user
            with gr.Group():
                gr.Markdown("Model Selection")
                self.model = gr.Dropdown(choices=[], label="Select a saved model for testing!", interactive=True)
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.bs_input = gr.Number(label="Batch Size")
            with gr.Group():
                gr.Markdown("Testing Dataset")
                self.test_path_input = gr.Textbox(label="Testing Folder Path", placeholder="/absolute/path/to/your/dataset")
            with gr.Group():
                gr.Markdown("Architecture")
                with gr.Row():
                    self.layer1_input = gr.Checkbox(label="Use Layer1")
                    self.layer2_input = gr.Checkbox(label="Use Layer2")
                    self.layer3_input = gr.Checkbox(label="Use Layer3")
                    self.layer4_input = gr.Checkbox(label="Use Layer4")
            
            with gr.Group():
                gr.Markdown("Refresh and Download")
                with gr.Column():
                    with gr.Row(equal_height=True):
                            self.refresh_btn = gr.Button("Refresh Saved Models")
                            self.refresh_btn.click(
                                fn=self.get_user_models,
                                inputs=[self.current_user],
                                outputs=[self.model]
                            )
                            self.download_btn = gr.Button("Download selected model")
                self.download_status = gr.Textbox(label="Download Status")

                self.download_btn.click(
                    fn=self.download_user_models,
                    inputs=[self.current_user, self.model],
                    outputs=[self.download_status]
                )
            with gr.Group():
                gr.Markdown("Testing")
                with gr.Column():
                    self.test_btn = gr.Button("Start Testing")
                    with gr.Row(equal_height=True):
                        self.test_status = gr.Textbox(label="Status")
                        self.test_graph = gr.Plot(label="Confusion Matrix")

                self.test_btn.click(
                fn=self.test_pipeline,
                inputs=[self.current_user, self.model, self.layer1_input, self.layer2_input, self.layer3_input, self.layer4_input, self.test_path_input, self.bs_input],
                outputs=[self.test_status, self.test_graph])

    def test_pipeline(self, username, model, layer1, layer2, layer3, layer4, test_folder, bs):

        path = test_folder.name if hasattr(test_folder, "name") else test_folder
        
        if not SecurityManager(path).validate_path():
            return "Invalid training folder", None
        
        test_path = os.path.join(path, "test")
        class_names = mm.test_transforms_dataset(None, test_path, bs)
        
        num_classes = len(class_names)
        loaded_model = mm.load_model(username, model, layer1, layer2, layer3, layer4, num_classes)
        
        test_metrics, all_labels, all_preds =  mm.test(loaded_model)
       
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        model_entry = users[username]["models"][model]
        model_entry["confusion_matrix"] = [
            [int(x) for x in all_labels],
            [int(x) for x in all_preds]
        ]
        model_entry["class_names"] = class_names

        with open(USER_DB, "w") as f:
            json.dump(users, f, indent=4)

        fig = gm.update_confusion_matrix(all_labels, all_preds, class_names)
        if NOTIFICATIONS_ENABLED:
            gr.Info("Testing completed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()

        return test_metrics, fig
    
    def get_user_models(self, username):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        model_names = list(users[username]["models"].keys())
        return gr.update(choices=model_names, value=None)
    
    def download_user_models(self, username, model):
        download_status = mm.download_model(username, model)
        if NOTIFICATIONS_ENABLED:
            gr.Info("Downloading completed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()
        return download_status
    
class LoginSignUp():
    def __init__(self):    
            self.current_user = gr.State(value=None)
            gr.Markdown("### Login / Sign Up.")
            with gr.Row(equal_height=True, elem_classes="spaced-row"):
                with gr.Group():
                    gr.Markdown("Login / Sign Up")
                    self.login_username = gr.Textbox(label="Username")
                    self.login_password = gr.Textbox(label="Password")
                    self.login_btn = gr.Button("Login")
                    self.login_status = gr.Textbox(label="Status", interactive=False)
                with gr.Group():
                    gr.Markdown("Sign Up")
                    self.signup_username = gr.Textbox(label="Username")
                    self.signup_password = gr.Textbox(label="Password")
                    self.signup_btn = gr.Button("Create Account")
                    self.signup_status = gr.Textbox(label="Status", interactive=False)

                    self.signup_btn.click(
                        fn=self.signup_pipeline,
                        inputs=[self.signup_username, self.signup_password],
                        outputs=self.signup_status
                    )

    @staticmethod
    def load_users():
        try:
            if not os.path.exists(USER_DB):
                return {}
            with open(USER_DB, "r") as f:
                return json.load(f)
        except Exception as e:
            return f"Exception: {e}"
        
    @staticmethod
    def save_users(users):
        try:
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
        except Exception as e:
            return f"Exception: {e}"

    @staticmethod
    def hash_password(password: str):
        try:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        except Exception as e:
            return f"Exception: {e}"

    @staticmethod
    def check_password(password: str, hashed: str):
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception as e:
            return f"Exception: {e}"

    def signup_pipeline(self, username, password):
        users = self.load_users()

        if not username or not password:
            return "Username and password required"

        if username in users:
            return "Username already exists"
        if NOTIFICATIONS_ENABLED:
            gr.Info("Sign Up completed successfully!", duration=8)
            users[username] = {
                "password": self.hash_password(password),
                "models": {}
            }
        if SOUNDSENABLED:
            pygame.mixer.music.play()


        self.save_users(users)
        return "Account created!"
    
    def login_pipeline(self, username, password):
        users = self.load_users()

        if username not in users:
            return "User not found", None
        
        if not self.check_password(password, users[username]["password"]):
            return "Incorrect password", None
        
        if NOTIFICATIONS_ENABLED:
            gr.Info("Log In completed successfully!", duration=5)
        if SOUNDSENABLED:
            pygame.mixer.music.play()
        return f"Welcome {username}!", username

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
            gr.Markdown("Architecture")
            with gr.Row():
                self.layer1_input = gr.Checkbox(label="Use Layer1")
                self.layer2_input = gr.Checkbox(label="Use Layer2")
                self.layer3_input = gr.Checkbox(label="Use Layer3")
                self.layer4_input = gr.Checkbox(label="Use Layer4")
        
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
            inputs=[self.current_user, self.image_path_input, self.model, self.layer1_input, self.layer2_input, self.layer3_input, self.layer4_input, self.bs_input],
            outputs=[self.predclass, self.originimage, self.gradimage]
        )

        self.augbutton.click(
            fn=self.update_augmentations,
            inputs=[self.image_path_input, self.transforms, self.bs_input],
            outputs=[self.augmentation]
        )

    def update_gradcam(self, username, dataset_path, model_name, layer1, layer2, layer3, layer4, bs):
        
        gradcam_path = os.path.join(dataset_path, "test")
        class_names = mm.test_transforms_dataset(None, gradcam_path, bs)
        num_classes = len(class_names)

        dataset = mm.test_dataset
        index = random.randint(0, len(dataset) - 1) 
        pil_image, _ = dataset[index]

        predicted_class, originimage, cam_image = mm.gradcam(username, pil_image, model_name, layer1, layer2, layer3, layer4, num_classes)
        predicted_label = class_names[predicted_class].capitalize()
        if NOTIFICATIONS_ENABLED:
            gr.Info("GradCAM computation completed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()
        return predicted_label, originimage, cam_image
    
    def update_augmentations(self, augmentation_path, selected_transforms, bs):
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
        if not aug_transforms:
            return f"No augmentation transforms supplied, at least one must be supplied."
        else:
            augmented_transforms = transforms.Compose(aug_transforms)

        augmentation_path = os.path.join(augmentation_path, "test")
        mm.test_transforms_dataset(augmented_transforms, augmentation_path, bs)

        dataset = mm.test_dataset
        index = random.randint(0, len(dataset) - 1) 
        pil_image, _ = dataset[index]
        if NOTIFICATIONS_ENABLED:
            gr.Info("Augmentated image computed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()

        return pil_image
    
    def get_user_models(self, username):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        model_names = list(users[username]["models"].keys())
        return gr.update(choices=model_names, value=None)
    
class Settings():
    def __init__(self, current_user):
        gr.Markdown("# Configure Settings")

        self.current_user = current_user

        with gr.Group():
            gr.Markdown("Music Player")

            self.audio_file = gr.Audio(
                type="filepath", 
                label="Upload & Play Music", 
                interactive=True
            )
        
        with gr.Group():
            gr.Markdown("Notifications and Sounds")
            
            self.notifications = gr.Checkbox(label="Enable Gradio Notifications", value=True, interactive=True)
            self.sounds = gr.Checkbox(label="Enable Sound Effects", value=True, interactive=True)
            self.volume_slider = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Sound Volume (%)", interactive=True)

        with gr.Group():
            gr.Markdown("Session")
            self.logout_btn = gr.Button("Log Out")
            self.close_btn = gr.Button("Close App")
        
        self.notifications.change(
            fn=self.toggle_notifications,
            inputs=[self.notifications],
            outputs=None
        )

        self.sounds.change(
            fn=self.toggle_sounds,
            inputs=[self.sounds],
            outputs=None
        )

        self.volume_slider.change(
            fn=self.update_volume,
            inputs=[self.volume_slider],
            outputs=None
        )    

        self.logout_btn.click(fn=self.logout)
        self.close_btn.click(fn=self.close_app)
        
    def toggle_notifications(self, notifications: bool):
        global NOTIFICATIONS_ENABLED
        NOTIFICATIONS_ENABLED = notifications
        return gr.Info(f"Notifications {'enabled' if notifications else 'disabled'}", duration=2)
    
    def toggle_sounds(self, sounds: bool):
        global SOUNDSENABLED
        SOUNDSENABLED = sounds
        return gr.Info(f"Sound effects {'enabled' if SOUNDSENABLED else 'disabled'}", duration = 2)

    def update_volume(self, volume: float):
        global CURRENTVOLUME
        CURRENTVOLUME = volume / 100.0
        if pygame.mixer.get_init():
            pygame.mixer.music.set_volume(CURRENTVOLUME)

    def logout(self):
        gr.Info(f"Logging out...", duration=3)
        return None

    def close_app(self):
        gr.Info(f"Closing application...", duration=3)

        def kill():
            time.sleep(2)
            os._exit(0)
        
        threading.Thread(target=kill).start()
        
        return
                    

        