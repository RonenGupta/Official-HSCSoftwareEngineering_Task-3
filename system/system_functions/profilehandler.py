import json
from system.backend_config.config import USER_DB

class ProfileManager():
    """Handles all operations regarding user profile preferences"""
    def __init__(self):
        # No initialisation needed, but class structure kept for clarity
        pass

    def get_preferences(self, username):
        """Gets the current user preferences"""
        
        # If username is a Gradio State object, extract the actual value
        if hasattr(username, "value"):
            username = username.value
        
        # If no username is provided, return default preferences
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
            # Load the user database JSON file
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            # Ensure the username exists
            if username not in users:
                raise ValueError("Username not found.")
            
            # Return stored preferences
            return users[username].get("preferences", {})
        
        # Error handling in case of a RuntimeError
        except Exception as e:
            raise RuntimeError(f"Failed to load preferences: {e}")

    def update_preference(self, username, key, value):
        """Updates a single preference value for a user"""

        # Extract username if wrapped in Gradio State
        if hasattr(username, 'value'):
            username = username.value
        
        # Username must be provided
        if not username:
            raise ValueError("No username provided.")
        
        try:
            # Load the user database
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            # Ensure the username exists
            if username not in users:
                raise ValueError("User not found.")
            
            # Ensure the user has a preferences section
            if "preferences" not in users[username]:
                users[username]["preferences"] = {}
                
            # Update the specific key
            users[username]["preferences"][key] = value

            # Save updated preferences back to json
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)

        # Error handling in case of a RuntimeError  
        except Exception as e:
            raise RuntimeError(f"Failed to update preferences: {e}")
