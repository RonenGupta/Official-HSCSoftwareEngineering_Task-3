import gradio as gr
from modelhandler import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager
from torchvision import transforms
import bcrypt
import os
import json
import random

USER_DB = "users.json"

mm = ModelManager()
gm= GraphManager()

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

            self.current_user = current_user
            with gr.Group():
                gr.Markdown("Dataset Input")
                self.train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate")
                self.epoch_input = gr.Number(label="Epochs")
                self.bs_input = gr.Number(label="Batch Size")
            with gr.Group():
                gr.Markdown("Transforms")
                self.train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")
            with gr.Group():
                gr.Markdown("Architecture")
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
            inputs=[self.train_path_input, self.epoch_input, self.lr_input, self.bs_input, self.layer4_input, self.train_transforms_input],
            outputs=[self.train_status, self.train_graph, self.acc_graph])

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
            final_transforms.append(transforms.RandomResizedCrop(224, 
                                                                scale=(0.6, 1.0)))
        
        if "RandomHorizontalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomHorizontalFlip(0.5))

        if "RandomVerticalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomVerticalFlip(0.5))
        
        if "RandomRotation" in selected_transforms:
            final_transforms.append(transforms.RandomRotation(360))

        if "ColorJitter" in selected_transforms:
            final_transforms.append(color_jitter = transforms.ColorJitter(brightness=0.5, 
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
        mm.build(layer4)

        gen =  mm.train(epochs, lr)
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
            gr.Info("Training completed successfully!", duration=8)

        self.trained_model = mm.model

    def save_model(self, user, model_name):
        if not hasattr(self, "trained_model"):
            return "No model to train"
        
        if not model_name:
            return "Must pass model name"
        gr.Info("Saving completed successfully!", duration=8)
        return mm.save_model(self.trained_model, user, model_name)

class Test_Tab():
    def __init__(self, current_user):
            gr.Markdown("### Test Models.")
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
                gr.Markdown("Model Selection")
                self.model = gr.Dropdown(choices=[], label="Select a saved model for testing!", interactive=True)
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.bs_input = gr.Number(label="Batch Size")
            with gr.Group():
                gr.Markdown("Testing Dataset")
                self.test_path_input = gr.Textbox(label="Testing Folder Path", placeholder="/absolute/path/to/your/dataset")
            with gr.Group():
                gr.Markdown("Transforms")
                self.test_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for testing!")
            with gr.Group():
                gr.Markdown("Architecture")
                self.layer4_input = gr.Checkbox(label="Use Layer4 (If you trained the model with layer 4, enable this)")
            
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
                inputs=[self.current_user, self.model, self.layer4_input, self.test_path_input, self.bs_input, self.test_transforms_input],
                outputs=[self.test_status, self.test_graph])

    def test_pipeline(self, username, model, layer4, test_folder, bs, selected_transforms):

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
        
        if "RandomVerticalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomVerticalFlip(0.5))

        if "RandomRotation" in selected_transforms:
            final_transforms.append(transforms.RandomRotation(360))
        
        if "ColorJitter" in selected_transforms:
            final_transforms.append(transforms.ColorJitter(
                                                            brightness=0.5, 
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
            test_transforms = None
        else:
            test_transforms = transforms.Compose(final_transforms)
        
        if not SecurityManager(path).validate_path():
            return "Invalid training folder", None
        
        test_path = os.path.join(path, "test")
        class_names = mm.test_transforms_dataset(test_transforms, test_path, bs)
        
        num_classes = len(class_names)
        loaded_model = mm.load_model(username, model, layer4, num_classes)
        
        test_metrics, all_labels, all_preds =  mm.test(loaded_model)
        fig = gm.update_confusion_matrix(all_labels, all_preds, class_names)
        gr.Info("Testing completed successfully!", duration=8)

        return test_metrics, fig
    
    def get_user_models(self, username):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        model_names = list(users[username]["models"].keys())
        return gr.update(choices=model_names, value=None)
    
    def download_user_models(self, username, model):
        download_status = mm.download_model(username, model)
        gr.Info("Downloading completed successfully!", duration=8)
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

                    self.login_btn.click(fn=self.login_pipeline,
                                            inputs=[self.login_username, self.login_password],
                                            outputs=[self.login_status, self.current_user]
                                            )
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
        
        gr.Info("Sign Up completed successfully!", duration=8)
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
        
        gr.Info("Log In completed successfully!", duration=8)
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
            self.transforms = gr.CheckboxGroup(choices=transform_options, label="Select transforms for testing!")
        
        with gr.Group():
            gr.Markdown("Hyperparameters")
            self.bs_input = gr.Number(label="Batch Size")
        with gr.Group():
            gr.Markdown("Architecture")
            self.layer4_input = gr.Checkbox(label="Use Layer4 (If you trained the model with layer 4, enable this)")
        
        with gr.Group():
            gr.Markdown("Refresh Models")
            self.refresh_btn = gr.Button("Refresh Saved Models")

        with gr.Group():
            gr.Markdown("View Augmentations")
            self.augbutton = gr.Button("View Augmentation Examples")
            self.augmentation = gr.Image(type="numpy")
        
        with gr.Group():
            gr.Markdown("GradCAM Results")
            self.gcbutton = gr.Button("Generate GradCAM")
            with gr.Row(equal_height=True):
                self.GradCAM = gr.Image(type="numpy")
                self.predclass = gr.Label()

        self.refresh_btn.click(
            fn=self.get_user_models,
            inputs=[self.current_user],
            outputs=[self.model]
        )
        
        self.gcbutton.click(
            fn=self.update_gradcam,
            inputs=[self.current_user, self.image_path_input, self.model, self.transforms, self.layer4_input, self.bs_input],
            outputs=[self.predclass, self.GradCAM]
        )

        self.augbutton.click(
            fn=self.update_augmentations,
            inputs=[self.image_path_input, self.transforms, self.bs_input],
            outputs=[self.augmentation]
        )

    def update_gradcam(self, username, dataset_path, model_name, selected_transforms, layer4, bs):
        final_transforms = []

        if "Resize(256, 256)" in selected_transforms:
            final_transforms.append(transforms.Resize((224, 224)))
        
        if "CenterCrop(224)" in selected_transforms:
            final_transforms.append(transforms.CenterCrop(224))
        
        if "RandomResizedCrop(224)" in selected_transforms:
            final_transforms.append(transforms.RandomResizedCrop(224, scale=(0.6, 1.0)))
        
        if "RandomHorizontalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomHorizontalFlip(0.5))

        if "RandomVerticalFlip" in selected_transforms:
            final_transforms.append(transforms.RandomVerticalFlip(0.5))

        if "RandomRotation" in selected_transforms:
            final_transforms.append(transforms.RandomRotation(360))

        if "ColorJitter" in selected_transforms:
            final_transforms.append(transforms.ColorJitter(
                                                        brightness=0.5, 
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

        if transforms.ToTensor() not in final_transforms:
            final_transforms.append(transforms.ToTensor())
            
        if not final_transforms:
            gradcam_transforms = None
        else:
            gradcam_transforms = transforms.Compose(final_transforms)
        
        gradcam_path = os.path.join(dataset_path, "test")
        class_names = mm.test_transforms_dataset(gradcam_transforms, gradcam_path, bs)
        num_classes = len(class_names)

        dataset = mm.test_dataset
        index = random.randint(0, len(dataset) - 1) 
        pil_image, _ = dataset[index]

        predicted_class, cam_image = mm.gradcam(username, pil_image, model_name, layer4, num_classes)
        predicted_label = class_names[predicted_class].capitalize()
        gr.Info("GradCAM computation completed successfully!", duration=8)
        return predicted_label, cam_image
    
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
        gr.Info("Augmentated image computed successfully!", duration=8)

        return pil_image
    
    def get_user_models(self, username):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        model_names = list(users[username]["models"].keys())
        return gr.update(choices=model_names, value=None)
    
class Settings():
    def __init__(self):
        gr.Markdown("# Configure Settings")


        with gr.Group():
            gr.Markdown("Text Size")

        with gr.Group():
            gr.Markdown("Music Player")

            self.audio_file = gr.Audio(
                type="filepath", 
                label="Upload & Play Music", 
                interactive=True
            )
                
        with gr.Group():
            gr.Markdown("Log Out")

        with gr.Group():
            gr.Markdown("Close App")

   

                    

        