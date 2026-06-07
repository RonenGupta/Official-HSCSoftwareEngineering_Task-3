import gradio as gr
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
import datetime
import re
import bcrypt
import os
import json
import pygame
from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER

pygame.mixer.init()
notification_sound = os.path.join(MUSIC_FOLDER, "ping.mp3")
pygame.mixer.music.load(notification_sound)

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

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
                    self.login_btn = gr.Button("Login", elem_id="auth_login_btn")
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
            raise RuntimeWarning(f"Failed to load users: {e}")
        
    @staticmethod
    def save_users(users):
        try:
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
        except Exception as e:
            raise RuntimeWarning(f"Failed to save users: {e}")

    @staticmethod
    def hash_password(password: str):
        try:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        except Exception as e:
            raise RuntimeWarning(f"Password hashing failed: {e}")

    @staticmethod
    def check_password(password: str, hashed: str):
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception as e:
            raise RuntimeWarning(f"Password check failed: {e}")
    
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
        try:
            users = self.load_users()

            if not self.validate_email(email):
                return gr.Warning("Invalid email format")
                
            if not username or len(username) < 3:
                return gr.Warning("Username must be at least 3 characters")

            if not self.validate_password(password):
                return gr.Warning("Password must be 8+ characters with letters and numbers")
            
            if password != confirm:
                return gr.Warning("Passwords do not match")
            
            if self.find_user(users, username):
                return gr.Warning("Username already exists")
            
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
                pygame.mixer.music.load(notification_sound)
                pygame.mixer.music.play()

            return "Account created sucessfully!", None
        
        except Exception as e:
            return gr.Warning(str(e))
        
    def login_pipeline(self, username, password):
        try:
            users = self.load_users()

            if not username or not password:
                return gr.Warning("Please enter both username and password"), None

            if self.failed_attempts.get(username, 0) >= 5:
                return gr.Warning("Too many failed attempts. Try again later."), None
            
            real_user = self.find_user(users, username)
            if not real_user:
                self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
                return gr.Warning("Invalid username or password"), None
            
            if not self.check_password(password, users[real_user]["password"]):
                self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
                return gr.Warning("Incorrect password"), None
            
            self.failed_attempts[username] = 0

            users[real_user]["last_login"] = str(datetime.datetime.now())
            self.save_users(users)

            if NOTIFICATIONS_ENABLED:
                gr.Info("Log In completed successfully!", duration=5)
            if SOUNDSENABLED:
                pygame.mixer.music.load(notification_sound)
                pygame.mixer.music.play()
            return f"Welcome {real_user}!", real_user
        except Exception as e:
            return gr.Warning(str(e)), None
    
    