import gradio as gr
from system.system_functions.securityhandler import SecurityManager
from system.system_functions.profilehandler import ProfileManager
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
import os
import json
import pygame

from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER

USER_DB = "users.json"

pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/ping.mp3"
pygame.mixer.music.load(music_path)

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

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
    
    def refresh_preferences(self, user):
        prefs = pm.get_preferences(user)
        return (
            gr.update(value=prefs.get("default_batch_size", 32)),
        )