import gradio as gr
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
import os
import json
import pygame
import time
import threading
import shutil

from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, CURRENTVOLUME, USER_DB, MUSIC_FOLDER


pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/LevinIntro.mp3"
pygame.mixer.music.load(music_path)

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

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

            default_activation_layers = prefs.get("default_activation_layer", "layer4")
            if isinstance(default_activation_layers, str):
                default_activation_layers = [default_activation_layers]

            self.pref_act_layer = gr.CheckboxGroup(
                choices=["layer1", "layer2", "layer3", "layer4"],
                label="Default Activation Layers",
                value=default_activation_layers
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
            pm.update_preference(username, "default_architecture", arch)
            pm.update_preference(username, "default_learning_rate", lr)
            pm.update_preference(username, "default_epochs", epochs)
            pm.update_preference(username, "default_batch_size", bs)
            pm.update_preference(username, "default_dropout", dropout)
            pm.update_preference(username, "default_activation_layer", act_layer)
            pm.update_preference(username, "default_featureviz_layer", fv_layer)
            pm.update_preference(username, "default_featureviz_channel", fv_channel)
            pm.update_preference(username, "notifications", notifications)
            pm.update_preference(username, "sound", sound)

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
            full_path = os.path.join(MUSIC_FOLDER, track)
            pygame.mixer.music.load(full_path)
            pygame.mixer.music.set_volume(CURRENTVOLUME)
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
                    

        