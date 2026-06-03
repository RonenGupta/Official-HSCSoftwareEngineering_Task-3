import gradio as gr
from modelhandler import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager
from profilehandler import ProfileManager
import torch
from torchvision import transforms
import threading
import time
import datetime
import re
import bcrypt
import os
import json
import random
import pygame
import shutil
from fpdf import FPDF

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device

USER_DB = "users.json"
MUSIC_FOLDER = "music"

pygame.mixer.init()
music_path = "/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/music/LevinIntro.mp3"
pygame.mixer.music.load(music_path)

NOTIFICATIONS_ENABLED = True
SOUNDSENABLED = True
CURRENTVOLUME = 0.5

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class Dashboard():
    def __init__(self, current_user):
        self.current_user = current_user
        gr.Markdown(
            f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <img src='/gradio_api/file=static/MyCNN.jpg' 
                     style='width: 800px; height: auto; display: block; margin-left: auto; margin-right: auto;' 
                     alt='MyCNN Logo' />
                <p style='font-size: 1.1rem; color: #555; margin-bottom: 10px;'>
                    Your central hub for training, testing, and managing convolutional models.
                </p>
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
                self.model_count = gr.Textbox(label="Saved Models", interactive=False)
            with gr.Column():
                self.last_model = gr.Textbox(label = "Last Trained Model", interactive=False)
            with gr.Column():
                self.last_accuracy = gr.Textbox(label="Last Accuracy", interactive=False)
            with gr.Column():
                self.last_time = gr.Textbox(label="Last Time Trained", interactive=False)
        
        with gr.Group(elem_classes = "models-section"):
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
                                    slot_notes = gr.Textbox(label="Notes", visible=False, interactive=True)
                                    save_notes_btn = gr.Button("Save Notes", visible=False)
                                    slot_loss = gr.Plot(visible=False)
                                    slot_acc = gr.Plot(visible=False)
                                    slot_cm = gr.Plot(visible=False)
                                    download_btn = gr.Button("Download", visible=False)
                                    delete_btn = gr.Button("Delete", visible=False)
                                    pdf_btn = gr.Button("Generate PDF", visible=False)
                                    self.card_slots.append((acc, slot_md, slot_notes, save_notes_btn, slot_loss, slot_acc, slot_cm, download_btn, delete_btn, pdf_btn))

        for i, slot in enumerate(self.card_slots):
            acc, slot_md, slot_notes, save_notes_btn, slot_loss, slot_acc, slot_cm, download_btn, delete_btn, pdf_btn = slot

            save_notes_btn.click(
                fn=self.update_notes,
                inputs=[self.current_user, gr.Textbox(value=f"model{i+1}", visible=False), slot_notes], 
                outputs=[]
            )

            download_btn.click(
                fn=self.download_user_models,
                inputs=[self.current_user, gr.Textbox(value=f"model{i+1}", visible=False)],
                outputs=[]
            )

            delete_btn.click(
                fn=self.delete_model,
                inputs=[self.current_user, gr.Textbox(value=f"model{i+1}", visible=False)],
                outputs=[]
            )

            pdf_btn.click(
                fn=self.generate_pdf,
                inputs=[self.current_user, gr.Textbox(value=f"model{i+1}", visible=False)]
            )

    def load_dashboard(self, username):
        if not username:
            return "<h3>Not logged in</h3>", 0, "-", "-"
        
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        pic = users[username].get("preferences", {}).get("profile_picture", None)

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
            f"""
            ## **Welcome {username}!**
            ### Here's your latest model activity.
            """,
            count,
            last_model,
            last_acc,
            last_time,
            models,
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
                gr.update(visible=False),
                gr.update(visible=False),             
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
            f"Architecture: {data.get('architecture')}\n"
        )

            base = i * 10

            updates[base] = gr.update(visible=True)

            updates[base+1] = gr.update(value=html, visible=True)
            
            updates[base+2] = gr.update(value=data.get("notes", "No notes"), visible=True)

            updates[base+3] = gr.update(visible=True)

            if data.get("loss_curve"):
                loss_fig = gm.update_loss(data["loss_curve"], len(data["loss_curve"]))
                updates[base + 4] = gr.update(value=loss_fig, visible=True)

            if data.get("accuracy_curve"):
                acc_fig = gm.update_accuracy(data["accuracy_curve"], len(data["accuracy_curve"]))
                updates[base + 5] = gr.update(value=acc_fig, visible=True)

            if data.get("confusion_matrix"):
                labels, preds = data["confusion_matrix"]
                class_names = data.get("class_names", [])
                cm_fig = gm.update_confusion_matrix(labels, preds, class_names)
                updates[base + 6] = gr.update(value=cm_fig, visible=True)

            updates[base+7] = gr.update(visible=True)
            updates[base+8] = gr.update(visible=True)
            updates[base+9] = gr.update(visible=True)

        return updates
    
    def get_card_components(self):
        return [c for slot in self.card_slots for c in slot]
    
    def delete_model(self, username, model_name):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        if model_name in users[username]["models"]:
            del users[username]["models"][model_name]
        
        model_path = f"models/{username}/{model_name}.pth"
        if os.path.exists(model_path):
            os.remove(model_path)
        
        with open(USER_DB, "w") as f:
            json.dump(users, f, indent=4)
        
        return

    def download_user_models(self, username, model):
        mm.download_model(username, model)
        if NOTIFICATIONS_ENABLED:
            gr.Info("Downloading completed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()
        return 
    
    def generate_pdf(self, username, model_name):
        with open(USER_DB) as f:
            users = json.load(f)
        
        data = users[username]["models"][model_name]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Model Report: {model_name}")
        for k, v in data.items():
            pdf.cell(0, 10, f"{k}: {v}", ln=True)
        
        path = f"reports/{username}_{model_name}.pdf"
        pdf.output(path)

        return 
    
    def update_notes(self, username, model_name, new_notes):
        with open(USER_DB, "r") as f:
            users = json.load(f)

        users[username]["models"][model_name]["notes"] = new_notes

        with open(USER_DB, "w") as f:
            json.dump(users, f, indent=4)
        
        return gr.Info("Notes updated!", duration=6)
    
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
            prefs = pm.get_preferences(current_user)
            self.final_accuracy = gr.State()
            self.final_loss = gr.State()
            self.final_epochs = gr.State()

            self.gpu_history = []
            self.cpu_history = []
            self.ram_history = []
    
            with gr.Group():
                gr.Markdown("Dataset Input")
                self.train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate", value=prefs.get("default_learning_rate"))
                self.epoch_input = gr.Number(label="Epochs", value=prefs.get("default_epochs"))
                self.bs_input = gr.Number(label="Batch Size", value=prefs.get("default_batch_size"))
                self.dropout_input = gr.Slider(0.0, 0.7, step=0.05, label="Dropout Rate", value=prefs.get("default_dropout"))
            with gr.Group():
                gr.Markdown("Early Stopping")
                self.earlystopping_input = gr.Checkbox(label="Enable Early Stopping", value=False)
                self.patience_input = gr.Slider(1, 20, step=1, label="Patience (Epochs)")
            with gr.Group():
                gr.Markdown("Transforms")
                self.train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")
            with gr.Group():
                gr.Markdown("Architecture")
                self.archtype_input = gr.Dropdown(choices=architecture_options, label="Select preferred model architecture (The smaller the dataset, the smaller the architecture)", value=prefs.get("default_architecture"))
                with gr.Row():
                    self.layer1_input = gr.Checkbox(label="Use Layer1", value = prefs.get("default_activation_layer") == "layer1")
                    self.layer2_input = gr.Checkbox(label="Use Layer2", value = prefs.get("default_activation_layer") == "layer2")
                    self.layer3_input = gr.Checkbox(label="Use Layer3", value = prefs.get("default_activation_layer") == "layer3")
                    self.layer4_input = gr.Checkbox(label="Use Layer4", value = prefs.get("default_activation_layer") == "layer4")
            
            with gr.Group():
                gr.Markdown("Training")
                self.train_btn = gr.Button("Start Training")
                with gr.Row(equal_height=True):
                    self.train_status = gr.Textbox(label="Status", lines=10)
                    self.train_graph = gr.Plot(label="Loss Curve")
                    self.acc_graph = gr.Plot(label="Accuracy Curve")
            with gr.Group():
                gr.Markdown("System Analytics")
                with gr.Row(equal_height = True):
                    self.gpu_plot = gr.Plot(label="GPU Usage (%)")
                    self.cpu_plot = gr.Plot(label = "CPU / RAM (%)")
                self.analytics_json = gr.JSON(label="Live Analytics")
            with gr.Group():
                gr.Markdown("Save Model")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        self.save_btn = gr.Button("Save Model")
                        self.save_model_name = gr.Textbox(label="Saved Model Name", placeholder="model1")
                        self.save_status = gr.Textbox(label="Save Status", interactive=False)

            self.train_btn.click(
            fn=self.train_pipeline,
            inputs=[self.train_path_input, self.epoch_input, self.lr_input, self.bs_input, self.layer1_input, self.layer2_input, self.layer3_input, self.layer4_input, self.train_transforms_input, self.earlystopping_input, self.patience_input, self.archtype_input, self.dropout_input],
            outputs=[self.train_status, self.train_graph, self.acc_graph, self.gpu_plot, self.cpu_plot, self.analytics_json])

            self.save_btn.click(
            fn=self.save_model,
            inputs=[self.current_user, self.save_model_name, self.final_accuracy, self.final_loss, self.final_epochs],
            outputs=[self.save_status])
            
    def train_pipeline(self, train_folder, epochs, lr, bs, layer1, layer2, layer3, layer4, selected_transforms, earlystopping, patience, arch_type, dropout):
        
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
        mm.build(arch_type, layer1, layer2, layer3, layer4, dropout)

        gen =  mm.train(earlystopping, patience, epochs, lr)

        try:

            while True:
                log, losses, accuracies, analytics = next(gen)
                self.gpu_history.append(analytics["gpu"]["load"] if analytics["gpu"] else 0)
                self.cpu_history.append(analytics["cpu"])
                self.ram_history.append(analytics["ram"])
                fig = gm.update_loss(losses, len(losses))
                acc_fig = gm.update_accuracy(accuracies, len(accuracies))
                gpu_fig = gm.update_gpu_plot(self.gpu_history)
                cpu_ram_fig = gm.update_cpu_ram_plot(self.cpu_history, self.ram_history)

                yield log, fig, acc_fig, gpu_fig, cpu_ram_fig, analytics

        except StopIteration as e:
            losses, accuracies = e.value
            self.gpu_history.append(analytics["gpu"]["load"] if analytics["gpu"] else 0)
            self.cpu_history.append(analytics["cpu"])
            self.ram_history.append(analytics["ram"])
            fig = gm.update_loss(losses, epochs)
            acc_fig = gm.update_accuracy(accuracies, epochs)
            gpu_fig = gm.update_gpu_plot(self.gpu_history)
            cpu_ram_fig = gm.update_cpu_ram_plot(self.cpu_history, self.ram_history)

            yield log, fig, acc_fig, gpu_fig, cpu_ram_fig, analytics

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
        
            prefs = pm.get_preferences(current_user)
            with gr.Group():
                gr.Markdown("Model Selection")
                self.model = gr.Dropdown(choices=[], label="Select a saved model for testing!", interactive=True)
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.bs_input = gr.Number(label="Batch Size", value=prefs.get("default_batch_size"))
            with gr.Group():
                gr.Markdown("Testing Dataset")
                self.test_path_input = gr.Textbox(label="Testing Folder Path", placeholder="/absolute/path/to/your/dataset")
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
            with gr.Group():
                gr.Markdown("Testing")
                with gr.Column():
                    self.test_btn = gr.Button("Start Testing")
                    with gr.Row(equal_height=True):
                        self.test_status = gr.Textbox(label="Status")
                        self.test_graph = gr.Plot(label="Confusion Matrix")

                self.test_btn.click(
                fn=self.test_pipeline,
                inputs=[self.current_user, self.model, self.test_path_input, self.bs_input],
                outputs=[self.test_status, self.test_graph])

    def test_pipeline(self, username, model, test_folder, bs):

        path = test_folder.name if hasattr(test_folder, "name") else test_folder
        
        if not SecurityManager(path).validate_path():
            return "Invalid training folder", None
        
        test_path = os.path.join(path, "test")
        class_names = mm.test_transforms_dataset(None, test_path, bs)
        
        num_classes = len(class_names)
        loaded_model = mm.load_model(username, model, num_classes)
        
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
    
    
class LoginSignUp():
    def __init__(self):    
            self.current_user = gr.State(value=None)
            self.failed_attempts = {}
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
                    self.signup_email = gr.Textbox(label="Email")
                    self.signup_username = gr.Textbox(label="Username")
                    self.signup_password = gr.Textbox(label="Password")
                    self.signup_confirm = gr.Textbox(label="Confirm Password", type="password")
                    self.signup_btn = gr.Button("Create Account")
                    self.signup_status = gr.Textbox(label="Status", interactive=False)

                    self.signup_btn.click(
                        fn=self.signup_pipeline,
                        inputs=[self.signup_email, self.signup_username, self.signup_password, self.signup_confirm],
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
    
    @staticmethod
    def validate_email(email):
        return re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email)
    
    @staticmethod
    def validate_password(password: str):
        return (
            len(password) >= 8 and
            any(character.isdigit() for character in password) and
            any(character.isalpha() for character in password)
        )

    @staticmethod
    def find_user(users, username):
        for u in users:
            if u.lower() == username.lower():
                return u
        return None

    def signup_pipeline(self, email, username, password, confirm):
        users = self.load_users()

        if not self.validate_email(email):
            return "Invalid email format"
        
        if not username or len(username) < 3:
            return "Username must be at least 3 characters"

        if not self.validate_password(password):
            return "Password must be 8+ characters with letters and numbers"
        
        if password != confirm:
            return "Passwords do not match"
        
        if self.find_user(users, username):
            return "Username already exists"
        
        users[username] = {
            "email": email,
            "password": self.hash_password(password),
            "models": {},
            "join_date": str(datetime.datetime.now()),
            "preferences": {
                "default_architecture": "ResNet18",
                "default_learning_rate": 0.001,
                "default_epochs": 10,
                "default_batch_size": 32,
                "default_dropout": 0.2,

                "default_activation_layer": "layer4",
                "default_featureviz_layer": "layer4",
                "default_featureviz_channel": 0,

                "notifications": True,
                "sound": True,
                "profile_picture": None
                }
        }

        self.save_users(users)

        if NOTIFICATIONS_ENABLED:
            gr.Info("Sign Up completed successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()

        return "Account created!"
    
    def login_pipeline(self, username, password):
        users = self.load_users()

        if self.failed_attempts.get(username, 0) >= 5:
            return "Too many failed attempts. Try again later.", None
        
        real_user = self.find_user(users, username)
        if not real_user:
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return "Invalid username or password", None
        
        if not self.check_password(password, users[username]["password"]):
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return "Incorrect password", None
        
        self.failed_attempts[username] = 0

        users[real_user]["last_login"] = str(datetime.datetime.now())
        if NOTIFICATIONS_ENABLED:
            gr.Info("Log In completed successfully!", duration=5)
        if SOUNDSENABLED:
            pygame.mixer.music.play()
        return f"Welcome {real_user}!", real_user

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
        
        gradcam_path = os.path.join(dataset_path, "test")
        class_names = mm.test_transforms_dataset(None, gradcam_path, bs)
        num_classes = len(class_names)

        dataset = mm.test_dataset
        index = random.randint(0, len(dataset) - 1) 
        pil_image, _ = dataset[index]

        predicted_class, originimage, cam_image = mm.gradcam(username, pil_image, model_name, num_classes)
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

class FeatureViz():
    def __init__(self, current_user):
        gr.Markdown("### Feature Visualization (Activation Maximization)")
        self.current_user = current_user
        
        prefs = pm.get_preferences(current_user)
        with gr.Group():
            gr.Markdown("Model Selection")
            self.model = gr.Dropdown(choices=[], label="Select a saved model", interactive=True)

        with gr.Group():
            gr.Markdown("Visualization Mode")
            self.vis_mode = gr.Radio(["Channel Visualization", "Activation Maps"], label="Visualization Mode")

        with gr.Group():
            gr.Markdown("Layer and Channel")
            self.layer_name = gr.Textbox(label="Layer name (e.g. layer4.1.conv2)", value=prefs.get("default_featureviz_layer"))
            self.channel_idx_input = gr.Number(label = "Channel Index", elem_id="ch_idx", value=prefs.get("default_featureviz_channel"))
            self.input_image = gr.Image(label="Input Image", type="pil", elem_id="img_in", visible=False)

        with gr.Group():
            self.img_size = gr.Slider(64, 256, value = 224, step = 16, label="Image Size", elem_id="img_size")
            self.steps = gr.Slider(20, 1000, value=80, step=10, label="Optimization steps", elem_id="steps")
            self.lr = gr.Slider(0.001, 0.5, value=0.1, step=0.01, label="Learning rate", elem_id="lr")

        with gr.Group():
            self.refresh_btn = gr.Button("Refresh Saved Models")
            self.run_btn = gr.Button("Generate Feature Visualization")
            self.output_image = gr.Image(type="numpy", label="Synthesized Feature Image", interactive=False)
        
        self.refresh_btn.click(
            fn=self.get_user_models,
            inputs=[self.current_user],
            outputs=[self.model]
        )

        self.vis_mode.change(
            fn=self.update_visibility,
            inputs=[self.vis_mode],
            outputs=[self.channel_idx_input, self.input_image, self.img_size, self.steps, self.lr]
        )

        self.run_btn.click(
            fn=self.run_feature_viz,
            inputs=[self.current_user, self.model, self.vis_mode, self.layer_name, self.channel_idx_input, self.input_image, self.img_size, self.steps, self.lr],
            outputs=[self.output_image]
        )

    def get_user_models(self, username):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        model_names = list(users[username]["models"].keys())
        return gr.update(choices=model_names, value=None)
    
    def run_feature_viz(self, username, model_name, mode, layer_name, channel_idx, input_image, img_size, steps, lr):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        model_entry = users[username]["models"][model_name]

        if model_entry.get("class_names") is not None:
            num_classes = len(model_entry["class_names"])
        else:
            num_classes = model_entry.get("num_classes", 2)

        loaded_model = mm.load_model(username, model_name, num_classes)
        mm.model = loaded_model.to(device)

        if mode == "Channel":
            img = mm.feature_visualization(
            layer_name = str(layer_name),
            channel_idx = int(channel_idx),
            img_size = int(img_size),
            steps = int(steps),
            lr = float(lr),
        )

        elif mode == "Activation Maps":
            if input_image is None:
                return gr.Error("Please upload an image for activation maps.")
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            img_tensor = preprocess(input_image).unsqueeze(0).to(device)

            activations = mm.get_activation_maps(mm.model, layer_name, img_tensor)
            grid_img = mm.activation_grid(activations)

            return grid_img

        if NOTIFICATIONS_ENABLED:
            gr.Info("Feature visualization generated successfully!", duration=8)
        if SOUNDSENABLED:
            pygame.mixer.music.play()

        return img
    
    def update_visibility(self, mode):
        if mode == "Channel Visualization":
            return [
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            ]
        elif mode == "Activation Maps":
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]
        else:
            return [gr.update(visible=True)] * 5

class Settings():
    def __init__(self, current_user):
        gr.Markdown("# Configure Settings")

        self.current_user = current_user
        
        prefs = pm.get_preferences(current_user)

        with gr.Group():
            gr.Markdown("Music Player")
            self.music_dropdown = gr.Dropdown(
                choices=self.list_music_files(),
                label="Select Music Track"
            )

            self.play_btn = gr.Button("Play")
            self.stop_btn = gr.Button("Stop")

            self.custom_audio_file = gr.Audio(
                type="filepath", 
                label="Upload & Play Custom Music", 
                interactive=True
            )
        
        with gr.Group():
            gr.Markdown("Notifications and Sounds")
            
            self.notifications = gr.Checkbox(label="Enable Gradio Notifications", interactive=True, value=prefs.get("notifications", True))
            self.sounds = gr.Checkbox(label="Enable Sound Effects", interactive=True, value=prefs.get("sound", True))
            self.volume_slider = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Sound Volume (%)", interactive=True)

        with gr.Group():
            gr.Markdown("Session")
            self.logout_btn = gr.Button("Log Out")
            self.close_btn = gr.Button("Close App")
        
        with gr.Group():
            gr.Markdown("Profile Preferences")

            self.pref_arch = gr.Dropdown(
                ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"],
                label="Default Architecture",
                value=prefs.get("default_architecture", "ResNet18")
            )
            self.pref_lr = gr.Slider(0.0001, 0.1, label="Default Learning Rate", value=prefs.get("default_learning_rate", 0.001))
            self.pref_epochs = gr.Number(label="Default Epochs", value=prefs.get("default_epochs", 10))
            self.pref_bs = gr.Number(label="Default Batch Size", value=prefs.get("default_batch_size", 32))
            self.pref_dropout = gr.Slider(0.0, 0.7, label="Default Dropout", value=prefs.get("default_dropout", 0.2))

            self.pref_act_layer = gr.Dropdown(
                ["layer1", "layer2", "layer3", "layer4"],
                label="Default Activation Layer",
                value=prefs.get("default_activation_layer", "layer4")
            )

            self.pref_fv_layer = gr.Dropdown(
                ["layer1", "layer2", "layer3", "layer4"],
                label = "Default FeatureViz Layer",
                value=prefs.get("default_featureviz_layer", "layer4")
            )

            self.pref_fv_channel = gr.Number(label="Default FeatureViz Channel", value=prefs.get("default_featureviz_channel", 0))
        
            self.save_prefs_button = gr.Button("Save Preferences")

        with gr.Group():
            gr.Markdown("Profile Picture Customisation")
            self.profile_pic_upload = gr.Image(
            label="Upload Profile Picture",
            type="filepath"
            )
            self.save_pic_btn = gr.Button("Save Profile Picture")

        
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

        self.play_btn.click(
            fn=self.play_music,
            inputs=[self.music_dropdown],
            outputs=[]
        )    

        self.stop_btn.click(
            fn=self.stop_music,
            inputs=[],
            outputs=[]
        )

        self.save_prefs_button.click(
            fn=self.save_preferences,
            inputs=[
                self.current_user,
                self.pref_arch,
                self.pref_lr,
                self.pref_epochs,
                self.pref_bs,
                self.pref_dropout,
                self.pref_act_layer,
                self.pref_fv_layer,
                self.pref_fv_channel,
                self.notifications,
                self.sounds,
            ],
            outputs=[]
        )

        self.save_pic_btn.click(
            fn=self.save_profile_picture,
            inputs=[self.current_user, self.profile_pic_upload],
            outputs=[]
        )

        self.logout_btn.click(fn=self.logout)
        self.close_btn.click(fn=self.close_app)
    
    def save_preferences(
            self, current_user_state, arch, lr, epochs, bs, dropout,
            act_layer, fv_layer, fv_channel,
            notifications, sound
    ):

        username = current_user_state.value if hasattr(current_user_state, "value") else current_user_state

        if not username:
            return gr.Info("Please log in before saving preferences.")
        try:
            pm.update_preference(self.current_user, "default_architecture", arch)
            pm.update_preference(self.current_user, "default_learning_rate", lr)
            pm.update_preference(self.current_user, "default_epochs", epochs)
            pm.update_preference(self.current_user, "default_batch_size", bs)
            pm.update_preference(self.current_user, "default_dropout", dropout)
            pm.update_preference(self.current_user, "default_activation_layer", act_layer)
            pm.update_preference(self.current_user, "default_featureviz_layer", fv_layer)
            pm.update_preference(self.current_user, "default_featureviz_channel", fv_channel)
            pm.update_preference(self.current_user, "notifications", notifications)
            pm.update_preference(self.current_user, "sound", sound)

            return gr.Info("Preferences Saved!")
        except Exception as e:
            return gr.Error(f"Failed to save preferences: {str(e)}")
    
    def save_profile_picture(self, username, image_path):
        if username is None:
            return gr.Info("Please log in before uploading an image.")
        if image_path is None:
            return gr.Info("No image uploaded.")
        
        os.makedirs("static/profile_pics", exist_ok = True)
        save_path = f"static/profile_pics/{username}.png"
        shutil.copy(image_path, save_path)

        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        users[username]["preferences"]["profile_picture"] = save_path
        with open(USER_DB, "w") as f:
            json.dump(users, f, indent=4)
        
        return gr.Info("Profile picture updated!")
    
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
    
    def play_music(self, track):
        try:
            pygame.mixer.music.load(os.path.join(MUSIC_FOLDER, track))
            pygame.mixer.music.play()
            return gr.Info(f"Playing: {track}")
        except Exception as e:
            return f"Error: {e}"
    
    def stop_music(self):
        pygame.mixer.music.stop()
        return gr.Info(f"Music stopped")

    @staticmethod
    def list_music_files():
        return [f for f in os.listdir(MUSIC_FOLDER) if f.endswith(".mp3")]

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
                    

        