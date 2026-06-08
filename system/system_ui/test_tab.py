import gradio as gr
from system.system_functions.securityhandler import SecurityManager
from system.system_functions.profilehandler import ProfileManager
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
import os
import json
import pygame

from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER

# Initialise sound system
pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/ping.mp3"
pygame.mixer.music.load(music_path)

# Instantiate managers
mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class Test_Tab():
    """UI + logic for testing saved models and generating confusion matrices."""
    def __init__(self, current_user):
            gr.Markdown("### Test Models.")
            self.current_user = current_user

            # Load user preferences (default batch size)
            prefs = pm.get_preferences(current_user)

            # Model selection dropdown
            with gr.Group():
                gr.Markdown("Model Selection")
                self.model = gr.Dropdown(choices=[], label="Select a saved model for testing!", interactive=True)

            # Hyperparameters
            with gr.Group():
                gr.Markdown("Hyperparameters")
                self.bs_input = gr.Number(label="Batch Size", value=prefs.get("default_batch_size"))

            # Dataset path input
            with gr.Group():
                gr.Markdown("Testing Dataset")
                self.test_path_input = gr.Textbox(label="Testing Folder Path", placeholder="/absolute/path/to/your/dataset")

            # Refresh saved models
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

            # Testing section
            with gr.Group():
                gr.Markdown("Testing")
                with gr.Column():
                    self.test_btn = gr.Button("Start Testing")
                    # Output: status + confusion matrix plot
                    with gr.Row(equal_height=True):
                        self.test_status = gr.Textbox(label="Status")
                        self.test_graph = gr.Plot(label="Confusion Matrix")

                # Connect test button to pipeline
                self.test_btn.click(
                fn=self.test_pipeline,
                inputs=[self.current_user, self.model, self.test_path_input, self.bs_input],
                outputs=[self.test_status, self.test_graph])

    def test_pipeline(self, username, model, test_folder, bs):
        """Runs the full testing pipeline, validating inputs, loading dataset, loading model,
        running evaluation, saves confusion matrix + class names to DB, returns metrics + confusion matrix figure."""
        try:
            # Handle Gradio File objects OR plain strings
            path = test_folder.name if hasattr(test_folder, "name") else test_folder
            
            # Validate inputs
            if not username:
                return gr.Warning("Please log in before testing."), None
            if not model:
                return gr.Warning("Please select a model to test."), None
            if not path:
                return gr.Warning("Please enter a valid path to your dataset."), None
            if bs is None or bs <= 0:
                return gr.Warning("Batch size must be a positive number."), None

            # Validate dataset path security
            SecurityManager(path).validate_path()
            
            # Load dataset (expects a /test folder)
            test_path = os.path.join(path, "test")
            class_names = mm.test_transforms_dataset(None, test_path, bs)
            num_classes = len(class_names)

            # Load model
            loaded_model = mm.load_model(username, model, num_classes)
            
            # Run evaluation
            test_metrics, all_labels, all_preds =  mm.test(loaded_model)

            # Save confusion matrix + class names to DB
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

             # Generate confusion matrix figure
            fig = gm.update_confusion_matrix(all_labels, all_preds, class_names)
            
            # Notifications
            if NOTIFICATIONS_ENABLED:
                gr.Info("Testing completed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.play()

            return test_metrics, fig
        except Exception as e:
            return gr.Warning(str(e)), None
    
    def get_user_models(self, username):
        """Load saved model names for dropdown"""
        try:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            model_names = list(users[username]["models"].keys())
            return gr.update(choices=model_names, value=None)
        except Exception as e:
            return gr.Warning(str(e))
    
    def refresh_preferences(self, user):
        """Refresh default batch size from user preferences."""
        prefs = pm.get_preferences(user)
        return (
            gr.update(value=prefs.get("default_batch_size", 32)),
        )