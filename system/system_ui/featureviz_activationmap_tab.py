import gradio as gr
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
from torchvision import transforms
import json
import pygame
from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER, device

pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/ping.mp3"
pygame.mixer.music.load(music_path)

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

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
            self.vis_mode = gr.Radio(["Channel Visualization", "Activation Maps"], label="Visualization Mode", value="Channel Visualization")

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
        try:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            model_names = list(users[username]["models"].keys())
            return gr.update(choices=model_names, value=None)
        except Exception as e:
            return gr.Warning(str(e))
    
    def refresh_preferences(self, user):
        prefs = pm.get_preferences(user)
        return (
            gr.update(value=prefs.get("default_featureviz_layer", "layer4")),
            gr.update(value=prefs.get("default_featureviz_channel", 0)),
        )
    
    def run_feature_viz(self, username, model_name, mode, layer_name, channel_idx, input_image, img_size, steps, lr):
        try:
            if not model_name:
                gr.Warning("Please select a model before generating GradCAM.")
                return None
            if not layer_name or layer_name.strip() == "":
                gr.Warning("Please enter a valid layer name.")
                return None
            
            with open(USER_DB, "r") as f:
                users = json.load(f)
            model_entry = users[username]["models"][model_name]

            if model_entry.get("class_names") is not None:
                num_classes = len(model_entry["class_names"])
            else:
                num_classes = model_entry.get("num_classes", 2)

            loaded_model = mm.load_model(username, model_name, num_classes)
            mm.model = loaded_model.to(device)

            if not mm.layer_exists(mm.model, layer_name):
                gr.Warning(f"Layer '{layer_name}' does not exist in the model.")
                return None

            if mode == "Channel Visualization":
                if img_size is None or int(img_size) <= 0:
                    gr.Warning("Image size must be positive")
                    return None
                if channel_idx is None or int(channel_idx) < 0:
                    gr.Warning("Channel index must be a non-negative integer.")
                    return None
                if steps is None or int(steps) <= 0:
                    gr.Warning("Steps must be positive.")
                    return None
                img = mm.feature_visualization(
                layer_name = str(layer_name),
                channel_idx = int(channel_idx),
                img_size = int(img_size),
                steps = int(steps),
                lr = float(lr),
            )
                if NOTIFICATIONS_ENABLED:
                    gr.Info("Feature visualization generated successfully!", duration=8)
                if SOUNDSENABLED:
                    pygame.mixer.music.play()
                return img

            elif mode == "Activation Maps":
                if input_image is None:
                    gr.Warning("Please upload an image for activation maps.")
                    return None
                
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

                img_tensor = preprocess(input_image).unsqueeze(0).to(device)

                activations = mm.get_activation_maps(mm.model, layer_name, img_tensor)
                grid_img = mm.activation_grid(activations)

                if NOTIFICATIONS_ENABLED:
                    gr.Info("Activation map generated successfully!", duration=8)
                if SOUNDSENABLED:
                    pygame.mixer.music.play()
                return grid_img
        except Exception as e:
            return gr.Warning(str(e))
    
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