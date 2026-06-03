import json
import gradio as gr

USER_DB = "users.json"

class ProfileManager():
    def __init__(self):
        pass

    def get_preferences(self, username):

        if isinstance(username, gr.State) or hasattr(username, "value"):
            username = username.value

        if not username:
            return {
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
            }
        try:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            return users.get(username, {}).get("preferences", {})
        except Exception:
            return {}

    def update_preference(self, username, key, value):
        if isinstance(username, gr.State) or hasattr(username, 'value'):
            username = username.value
        
        if not username:
            return
        
        with open(USER_DB, "r") as f:
            users = json.load(f)
        
        if "preferences" not in users[username]:
            users[username]["preferences"] = {}
            
        users[username]["preferences"][key] = value

        with open(USER_DB, "w") as f:
            json.dump(users, f, indent=4)
        return f"{key} updated!"
