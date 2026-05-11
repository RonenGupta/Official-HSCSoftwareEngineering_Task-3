import gradio as gr
from modelhandler import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager
from torchvision import transforms
import bcrypt
import os
import json

USER_DB = "users.json"

class Train_Tab():
    def __init__(self, current_user):
        
        with gr.Tab("Train"):
            gr.Markdown("# Training Demo")
            transform_options = [
                "Resize(256, 256)",
                "CenterCrop(224)",
                "RandomResizedCrop(224)",
                "RandomHorizontalFlip",
                "ToTensor",
                "Normalize"
            ]

            self.current_user = current_user
            self.lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate")
            self.epoch_input = gr.Number(label="Epochs")
            self.bs_input = gr.Number(label="Batch Size")
            self.train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")
            self.train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")
            self.layer4_input = gr.Checkbox(label="Use Layer4")
            with gr.Row(equal_height=True):
                with gr.Column():
                    self.save_model_name = gr.Textbox(label="Saved Model Name", placeholder="model1")
                    self.save_btn = gr.Button("Save Model")
                self.save_status = gr.Textbox(label="Save Status", interactive=False)
                self.train_btn = gr.Button("Start Training")
            with gr.Row(equal_height=True):
                self.train_status = gr.Textbox(label="Status", interactive=False)
                self.train_graph = gr.Plot(label="Loss Curve")

            self.train_btn.click(
            fn=self.train_pipeline,
            inputs=[self.train_path_input, self.epoch_input, self.lr_input, self.bs_input, self.layer4_input, self.train_transforms_input],
            outputs=[self.train_status, self.train_graph])

            self.save_btn.click(
            fn=self.save_model,
            inputs=[self.current_user, self.save_model_name],
            outputs=[self.save_status])
        
    def train_pipeline(self, train_folder, epochs, lr, bs, layer4, selected_transforms):
        
        path = train_folder.name if hasattr(train_folder, "name") else train_folder

        final_transforms = []

        if "Resize(256, 256)" in selected_transforms:
            final_transforms.append(transforms.Resize((224, 224)))
        
        if "CenterCrop(224)" in selected_transforms:
            final_transforms.append(transforms.CenterCrop(224))
        
        if "RandomResizedCrop(224)" in selected_transforms:
            final_transforms.append(transforms.RandomResizedCrop(224, scale=(0.6, 1.0)))
        
        if "RandomHorizontalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomHorizontalFlip(0.5))

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
        
        mm = ModelManager()
        mm.train_transforms_dataset(train_transforms, path, bs)
        mm.build(layer4)

        gen =  mm.train(epochs, lr)
        try:
            while True:
                log = next(gen)
                yield log, None
        except StopIteration as e:
            losses = e.value
            gm = GraphManager(losses, epochs)
            fig = gm.update_loss()

        self.trained_model = mm.model

        yield log, fig

    def save_model(self, user, model_name):
        if not hasattr(self, "trained_model"):
            return "No model to train"
        
        if not model_name:
            return "Must pass model name"
        
        mm = ModelManager()
        return mm.save_model(self.trained_model, user, model_name)

class Test_Tab():
    def __init__(self):
        
        with gr.Tab("Test"):
            gr.Markdown("# Testing Demo")
            transform_options = [
                "Resize(256, 256)",
                "CenterCrop(224)",
                "RandomResizedCrop(224)",
                "RandomHorizontalFlip",
                "ToTensor",
                "Normalize"
            ]
            self.model = gr.Dropdown(choices=["model1", "model2", "model3"], label="Select a saved model for testing!", interactive=True)
            self.bs_input = gr.Number(label="Batch Size")
            self.test_path_input = gr.Textbox(label="Testing Folder Path", placeholder="/absolute/path/to/your/dataset")
            self.test_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for testing!")
            self.test_btn = gr.Button("Start Training")
            with gr.Row(equal_height=True):
                self.test_status = gr.Textbox(label="Status")
                self.test_graph = gr.Plot(label="Confusion Matrix")

            self.test_btn.click(
            fn=self.test_pipeline,
            inputs=[self.test_path_input, self.bs_input, self.test_transforms_input],
            outputs=[self.test_status, self.test_graph])

    def test_pipeline(self, model, test_folder, bs, selected_transforms):

        path = test_folder.name if hasattr(test_folder, "name") else test_folder

        final_transforms = []

        if "Resize(256, 256)" in selected_transforms:
            final_transforms.append(transforms.Resize((224, 224)))
        
        if "CenterCrop(224)" in selected_transforms:
            final_transforms.append(transforms.CenterCrop(224))
        
        if "RandomResizedCrop(224)" in selected_transforms:
            final_transforms.append(transforms.RandomResizedCrop(224, scale=(0.6, 1.0)))
        
        if "RandomHorizontalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomHorizontalFlip(0.5))

        if "ToTensor" in selected_transforms:
            final_transforms.append(transforms.ToTensor())
        
        if "Normalize" in selected_transforms:
            final_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))
            
        if not final_transforms:
            test_transforms = None
        else:
            test_transforms = transforms.Compose(final_transforms)
        
        if not SecurityManager(path).validate_path():
            return "Invalid training folder", None
        
        mm = ModelManager()
        class_names = mm.test_transforms_dataset(test_transforms, path, bs)

        test_metrics, all_labels, all_preds =  mm.test(model)
        gm = GraphManager(None, None)
        fig = gm.update_confusion_matrix(all_labels, all_preds, class_names)

        return test_metrics, fig

class LoginSignUp():
    def __init__(self):    
        self.current_user = gr.State(value=None)

        with gr.Tab("Login / Sign Up"):
            gr.Markdown("Login / Sign Up")
            self.login_username = gr.Textbox(label="Username")
            self.login_password = gr.Textbox(label="Password")
            self.login_btn = gr.Button("Login")
            self.login_status = gr.Textbox(label="Status", interactive=False)

            self.login_btn.click(fn=self.login_pipeline,
                                 inputs=[self.login_username, self.login_password],
                                 outputs=[self.login_status, self.current_user]
                                 )
            gr.Markdown("---")
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
        
        users[username] = {
            "password": self.hash_password(password),
            "models": {}
        }

        self.save_users(users)
        return "Account created!"
    
    def login_pipeline(self, username, password):
        users = self.load_users()

        if username not in users:
            return "User not found", None
        
        if not self.check_password(password, users[username]["password"]):
            return "Incorrect password", None
        
        return f"Welcome {username}!", username
