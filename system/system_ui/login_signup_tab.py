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

# Initialise pygame for notification sounds
pygame.mixer.init()
notification_sound = os.path.join(MUSIC_FOLDER, "ping.mp3")
pygame.mixer.music.load(notification_sound)

# Instantiate system managers
mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class LoginSignUp():
    """Handles user authentication: login, signup, password hashing, validation."""
    def __init__(self):
            # Stores the currently logged-in user    
            self.current_user = gr.State(value=None)
            
            # Track failed login attempts to prevent brute-force attacks
            self.failed_attempts = {}

            # Header
            gr.Markdown("### Login / Sign Up.")

            # UI Layout: Login + Signup side-by-side
            with gr.Row(equal_height=True, elem_classes="spaced-row"):

                # Login Section
                with gr.Group():
                    gr.Markdown("Login / Sign Up")
                    self.login_username = gr.Textbox(label="Username")
                    self.login_password = gr.Textbox(label="Password")
                    self.login_btn = gr.Button("Login", elem_id="auth_login_btn")
                    self.login_status = gr.Textbox(label="Status", interactive=False)

                # Signup Section
                with gr.Group():
                    gr.Markdown("Sign Up")
                    self.signup_email = gr.Textbox(label="Email")
                    self.signup_username = gr.Textbox(label="Username")
                    self.signup_password = gr.Textbox(label="Password")
                    self.signup_confirm = gr.Textbox(label="Confirm Password", type="password")
                    self.signup_btn = gr.Button("Create Account")
                    self.signup_status = gr.Textbox(label="Status", interactive=False)

                    # Connect signup button to signup pipeline
                    self.signup_btn.click(
                        fn=self.signup_pipeline,
                        inputs=[self.signup_email, self.signup_username, self.signup_password, self.signup_confirm],
                        outputs=self.signup_status
                    )

    # User DB Helpers
    @staticmethod
    def load_users():
        """Load user database from JSON file."""
        try:
            if not os.path.exists(USER_DB):
                return {}
            with open(USER_DB, "r") as f:
                return json.load(f)
        
        # Error Handling in case of RuntimeWarning
        except Exception as e:
            raise RuntimeWarning(f"Failed to load users: {e}")
        
    @staticmethod
    def save_users(users):
        """Save updated user database to JSON file."""
        try:
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
        # Error Handling in case of RuntimeWarning
        except Exception as e:
            raise RuntimeWarning(f"Failed to save users: {e}")

    @staticmethod
    def hash_password(password: str):
        """Hash a password using bcrypt."""
        try:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        # Error Handling in case of RuntimeWarning
        except Exception as e:
            raise RuntimeWarning(f"Password hashing failed: {e}")

    @staticmethod
    def check_password(password: str, hashed: str):
        """Verify a password against its bcrypt hash."""
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        # Error Handling in case of RuntimeWarning
        except Exception as e:
            raise RuntimeWarning(f"Password check failed: {e}")
    
    @staticmethod
    def validate_email(email):
        """Validate email format using regex."""
        return re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email)
    
    @staticmethod
    def validate_password(password: str):
        """Password must be 8+ chars, contain letters + numbers."""
        return (
            len(password) >= 8 and
            any(character.isdigit() for character in password) and
            any(character.isalpha() for character in password)
        )

    @staticmethod
    def find_user(users, username):
        """Case-insensitive username lookup."""
        for u in users:
            if u.lower() == username.lower():
                return u
        return None

    def signup_pipeline(self, email, username, password, confirm):
        """Create a new user account with validation + hashing."""
        try:
            users = self.load_users()

            # Validate email
            if not self.validate_email(email):
                return gr.Warning("Invalid email format")
            
            # Validate username
            if not username or len(username) < 3:
                return gr.Warning("Username must be at least 3 characters")
            
            # Validate password strength
            if not self.validate_password(password):
                return gr.Warning("Password must be 8+ characters with letters and numbers")
            
            # Confirm password match
            if password != confirm:
                return gr.Warning("Passwords do not match")
            
            # Check if username already exists
            if self.find_user(users, username):
                return gr.Warning("Username already exists")
            
            # Create new user entry
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

            # Save updated DB
            self.save_users(users)

            # Notifications
            if NOTIFICATIONS_ENABLED:
                gr.Info("Sign Up completed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.load(notification_sound)
                pygame.mixer.music.play()

            return "Account created sucessfully!", None
        
        # Error Handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e))
        
    def login_pipeline(self, username, password):
        """Authenticate user with password verification + brute-force protection."""
        try:
            users = self.load_users()

            # Require both fields
            if not username or not password:
                return gr.Warning("Please enter both username and password"), None

            # Lockout after 5 failed attempts
            if self.failed_attempts.get(username, 0) >= 5:
                return gr.Warning("Too many failed attempts. Try again later."), None
            
            # Case-insensitive username lookup
            real_user = self.find_user(users, username)
            if not real_user:
                self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
                return gr.Warning("Invalid username or password"), None
            
            # Verify password
            if not self.check_password(password, users[real_user]["password"]):
                self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
                return gr.Warning("Incorrect password"), None
            
            # Reset failed attempts on success
            self.failed_attempts[username] = 0

            # Updates last login timestamp
            users[real_user]["last_login"] = str(datetime.datetime.now())
            self.save_users(users)

            # Notifications
            if NOTIFICATIONS_ENABLED:
                gr.Info("Log In completed successfully!", duration=5)
            if SOUNDSENABLED:
                pygame.mixer.music.load(notification_sound)
                pygame.mixer.music.play()
            return f"Welcome {real_user}!", real_user
        # Error Handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e)), None
    
    