import gradio as gr
import os
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
from torchvision import transforms
import json
import pygame
from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER, device

# Initialise audio system for notification sounds
pygame.mixer.init()
music_path = os.path.join(MUSIC_FOLDER, "ping.mp3")
pygame.mixer.music.load(music_path)

# Instantiate system managers
mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class FeatureViz():
    """UI and logic for Feature Visualization (Activation Maximization + Activation Maps)."""
    def __init__(self, current_user):
        # Display section header
        gr.Markdown("### Feature Visualization (Activation Maximization)")

        self.current_user = current_user
        
        # Load user preferences (default layer, default channel)
        prefs = pm.get_preferences(current_user)

        # Model selection dropdown
        with gr.Group():
            gr.Markdown("Model Selection")
            self.model = gr.Dropdown(choices=[], label="Select a saved model", interactive=True)

        # Visualization mode selector
        with gr.Group():
            gr.Markdown("Visualization Mode")
            self.vis_mode = gr.Radio(["Channel Visualization", "Activation Maps"], label="Visualization Mode", value="Channel Visualization")

        # Layer + channel inputs
        with gr.Group():
            gr.Markdown("Layer and Channel")
            self.layer_name = gr.Textbox(label="Layer name (e.g. layer4.1.conv2)", value=prefs.get("default_featureviz_layer"))
            self.channel_idx_input = gr.Number(label = "Channel Index", elem_id="ch_idx", value=prefs.get("default_featureviz_channel"))
            with gr.Group() as self.input_image_group:
                self.input_image = gr.Image(label="Input Image", type="pil", elem_id="img_in", visible=True)

        # Optimization hyperparameters
        with gr.Group():
            self.img_size = gr.Slider(64, 256, value = 224, step = 16, label="Image Size", elem_id="img_size")
            self.steps = gr.Slider(20, 1000, value=80, step=10, label="Optimization steps", elem_id="steps")
            self.lr = gr.Slider(0.001, 0.5, value=0.1, step=0.01, label="Learning rate", elem_id="lr")

        # Buttons + output image
        with gr.Group():
            self.refresh_btn = gr.Button("Refresh Saved Models")
            self.run_btn = gr.Button("Generate Feature Visualization")
            self.output_image = gr.Image(type="numpy", label="Synthesized Feature Image", interactive=False)
        
        # Refresh saved model list
        self.refresh_btn.click(
            fn=self.get_user_models,
            inputs=[self.current_user],
            outputs=[self.model]
        )

        # Change visibility when switching modes
        self.vis_mode.change(
            fn=self.update_visibility,
            inputs=[self.vis_mode],
            outputs=[self.channel_idx_input, self.input_image_group, self.img_size, self.steps, self.lr]
        )

        # Run feature visualization
        self.run_btn.click(
            fn=self.run_feature_viz,
            inputs=[self.current_user, self.model, self.vis_mode, self.layer_name, self.channel_idx_input, self.input_image, self.img_size, self.steps, self.lr],
            outputs=[self.output_image]
        )

    def get_user_models(self, username):
        """Load saved models for dropdown"""
        try:
            # Open user DB
            with open(USER_DB, "r") as f:
                users = json.load(f)

            # Get model names and update model dropdown
            model_names = list(users[username]["models"].keys())
            return gr.update(choices=model_names, value=None)
        
        # Error handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e))
    
    def refresh_preferences(self, user):
        """Refresh default preferences (layer + channel)"""
        # Get preferences for user
        prefs = pm.get_preferences(user)

        # Return updates for layer and channel index
        return (
            gr.update(value=prefs.get("default_featureviz_layer", "layer4")),
            gr.update(value=prefs.get("default_featureviz_channel", 0)),
        )
    
    def run_feature_viz(self, username, model_name, mode, layer_name, channel_idx, input_image, img_size, steps, lr):
        """Main feature visualization logic"""
        try:
            # Validate model selection
            if not model_name:
                gr.Warning("Please select a model before generating GradCAM.")
                return None
            # Validate layer name
            if not layer_name or layer_name.strip() == "":
                gr.Warning("Please enter a valid layer name.")
                return None
            
            # Load user DB entry for this model
            with open(USER_DB, "r") as f:
                users = json.load(f)
            model_entry = users[username]["models"][model_name]

            # Determine number of classes
            if model_entry.get("class_names") is not None:
                num_classes = len(model_entry["class_names"])
            else:
                num_classes = model_entry.get("num_classes", 2)

            # Load model into memory
            loaded_model = mm.load_model(username, model_name, num_classes)
            mm.model = loaded_model.to(device)

            # Validate layer exists
            if not mm.layer_exists(mm.model, layer_name):
                gr.Warning(f"Layer '{layer_name}' does not exist in the model.")
                return None

            # Feature visualization pipeline
            if mode == "Channel Visualization":

                # Validate inputs
                if img_size is None or int(img_size) <= 0:
                    gr.Warning("Image size must be positive")
                    return None
                if channel_idx is None or int(channel_idx) < 0:
                    gr.Warning("Channel index must be a non-negative integer.")
                    return None
                if steps is None or int(steps) <= 0:
                    gr.Warning("Steps must be positive.")
                    return None
                
                # Run activation maximization
                img = mm.feature_visualization(
                layer_name = str(layer_name),
                channel_idx = int(channel_idx),
                img_size = int(img_size),
                steps = int(steps),
                lr = float(lr),
            )
                # Notifications
                if NOTIFICATIONS_ENABLED:
                    gr.Info("Feature visualization generated successfully!", duration=8)
                if SOUNDSENABLED:
                    pygame.mixer.music.play()

                return img

            # Activation Map pipeline
            elif mode == "Activation Maps":

                # Must upload an image
                if input_image is None:
                    gr.Warning("Please upload an image for activation maps.")
                    return None
                
                # Preprocessing pipeline
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])

                # Convert image to tensor
                img_tensor = preprocess(input_image).unsqueeze(0).to(device)

                # Extract activation maps
                activations = mm.get_activation_maps(mm.model, layer_name, img_tensor)

                # Convert to grid image
                grid_img = mm.activation_grid(activations)

                # Notifications and sounds
                if NOTIFICATIONS_ENABLED:
                    gr.Info("Activation map generated successfully!", duration=8)
                if SOUNDSENABLED:
                    pygame.mixer.music.play()
                
                # Return grid image
                return grid_img
        
        # Error handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e))
    
    def update_visibility(self, mode):
        """Update visibility for channel_idx_input, input_image, img_size, steps, lr"""
        if mode == "Channel Visualization":
            # Hide input image (index 1), show everything else
            return [gr.update(visible=False) if i == 1 else gr.update(visible = True)
                    for i in range(5)]

        elif mode == "Activation Maps":
            # Show input image (index 1), hide everything else
            return [gr.update(visible=True) if i == 1 else gr.update(visible=False)
                for i in range(5)]
        # Fallback
        else:
            return [gr.update(visible=True)] * 5