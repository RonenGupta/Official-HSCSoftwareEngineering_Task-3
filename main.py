import gradio as gr
from pathlib import Path
import json
import pygame

# Import all UI tabs
from system.system_ui.assistant_acc import MyCNN_Assistant
from system.system_ui.dashboard_tab import Dashboard
from system.system_ui.train_tab import Train_Tab
from system.system_ui.test_tab import Test_Tab
from system.system_ui.gradcam_tab import GradCAM
from system.system_ui.featureviz_activationmap_tab import FeatureViz
from system.system_ui.login_signup_tab import LoginSignUp
from system.system_ui.settings_tab import Settings

# Initialize audio system
pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)

# Load user DB
USER_DB = 'users.json'

# Load CSS
css_path = Path("static/styles.css").read_text()

# Allow Gradio to serve static files (images, CSS, etc.)
gr.set_static_paths(paths=[Path.cwd() / "static"])

# Main Application Layout
with gr.Blocks(fill_height=True, fill_width=True) as demo:

    # Global applications state
    app_state = gr.State({
    "current_tab": "login",
    "user": None,
    "last_error": None,
    "model_status": None,
    "dashboard_info": {}
    })

    # Whether the assistant panel is visible or not
    assistant_state = gr.State(True)

    # Left Side: All main tabs (Login, Dashboard, Train, Test, GradCAM, FeatureViz, Settings)
    with gr.Row():
        with gr.Column(scale=8):

             # LOGIN TAB (visible by default)
            with gr.Group(elem_classes="animate__animated animate__fadeInLeft") as login_tab:
                login = LoginSignUp()

            # DASHBOARD TAB (hidden initially)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as dashboard_tab:
                dashboard = Dashboard(login.current_user)
            
            # TRAIN TAB
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as train_tab:
                train = Train_Tab(login.current_user)
            
            # TEST TAB
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as test_tab:
                test = Test_Tab(login.current_user)

            # GRADCAM TAB
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as gradcam_tab:
                gradcam = GradCAM(login.current_user)
            
            # SETTINGS TAB
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as settings_tab:
                settings = Settings(login.current_user)
            
            # FEATURE VIZ TAB
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as featureviz_tab:
                featureviz = FeatureViz(login.current_user)

        # Right Side: AI Assistant Panel
        cnn_assistant = MyCNN_Assistant()
        chat_panel, acc, assistant_chatbot, assistant_msg, audio_out = cnn_assistant.build_ui()

        # When user submits a message to the assistant
        assistant_msg.submit(
            fn=cnn_assistant.respond,
            inputs=[assistant_msg, assistant_chatbot, app_state],
            outputs=[assistant_msg, audio_out, assistant_chatbot]
        )

    # Tab Switching Functions
    def show_login(status, user, app_state_dict):
        """Login Tab"""
        app_state_dict["current_tab"] = "login"
        app_state_dict["user"] = user
        # Show login tab, hide all others
        return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
            if i == 0 else gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
            for i in range(7)],
            app_state_dict
        )

    def show_dashboard(status, user, app_state_dict):
        """Dashboard Tab"""
        app_state_dict["current_tab"] = "dashboard"
        app_state_dict["user"] = user
        
         # If not logged in, redirect to login
        with open(USER_DB, "r") as f:
            users = json.load(f)

        if not user or user not in users:
            gr.Warning("Please login before accessing other tabs.", duration=6)
            return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")   
            if i == 0 else gr.update(elem_classes="hidden-tab")
            for i in range(7)],  
            *["", "", "", "", ""],                    
            gr.update(value=None),                 
            *[gr.update(visible=False) for _ in dashboard.get_card_components()],
            app_state_dict
        )

        if user:
            # Load dashboard data
            welcome, count, last_model, last_acc, last_time, models = dashboard.load_dashboard(user)
            card_updates = dashboard.build_model_cards(models)
            try:
                # Load profile picture
                with open(USER_DB, "r") as f:
                    users = json.load(f)
            except Exception:
                return gr.Warning("User database is corrupted. Please reset users.json.")
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            # Show dashboard tab
            return (
                *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
                if i == 1 else gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
                for i in range(7)],
                welcome,
                count,
                last_model,
                last_acc,
                last_time,
                gr.update(value=pic),
                *card_updates,
                app_state_dict
            )
        return show_login(status, user, app_state_dict)

    def show_train(status, user, app_state_dict):
        """Train Tab"""
        app_state_dict["current_tab"] = "train"
        app_state_dict["user"] = user

        with open(USER_DB, "r") as f:
            users = json.load(f)

        if not user or user not in users:
            gr.Warning("Please login before accessing other tabs.", duration=6)

            return (
                *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
                if i == 0 else gr.update(elem_classes="hidden-tab")
                for i in range(7)],
                *[gr.update(value=None)
                for i in range(10)],
                app_state_dict
        )

        if user:

            # Load profile picture + preferences
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)
            pref_updates = train.refresh_preferences(user)

            return (
                *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
                  if i == 2 else gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
                  for i in range(7)],
                gr.update(value=pic),
                *pref_updates,
                app_state_dict
            )
        return show_login(status, user)

    
    def show_test(status, user, app_state_dict):
        """Test Tab"""
        app_state_dict["current_tab"] = "test"
        app_state_dict["user"] = user

        with open(USER_DB, "r") as f:
            users = json.load(f)
        if not user or user not in users:
            gr.Warning("Please login before accessing other tabs.", duration=6)

            return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
              if i == 0 else gr.update(elem_classes="hidden-tab")
              for i in range(7)],
              gr.update(value=None),
              gr.update(value=None),
              app_state_dict
        )
        if user:

            # Load profile picture + preferences
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)
            pref_updates = test.refresh_preferences(user)

            return (
                *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
                  if i == 3 else gr.update(elem_classes="hidden-tab")
                  for i in range(7)],
                gr.update(value=pic),
                *pref_updates,
                app_state_dict
            )
        return show_login(status, user)
    
    def show_gradcam(status, user, app_state_dict):
        """GRADCAM TAB"""
        app_state_dict["current_tab"] = "gradcam"
        app_state_dict["user"] = user
        with open(USER_DB, "r") as f:
            users = json.load(f)

        if not user or user not in users:
            gr.Warning("Please login before accessing other tabs.", duration=6)

            return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
              if i == 0 else gr.update(elem_classes="hidden-tab")
              for i in range(7)],
            gr.update(value=None),
            app_state_dict
        )
        if user:

            # Load profile picture + preferences
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
                if i == 4 else gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
                for i in range(7)],
                gr.update(value=pic),
                app_state_dict
            )
        return show_login(status, user)
    
    def show_featureviz(status, user, app_state_dict):
        """FEATURE VIZ TAB"""
        app_state_dict["current_tab"] = "featureviz"
        app_state_dict["user"] = user
        with open(USER_DB, "r") as f:
            users = json.load(f)

        if not user or user not in users:
            gr.Warning("Please login before accessing other tabs.", duration=6)

            return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
            if i == 0 else gr.update(elem_classes="hidden-tab")
            for i in range(7)],
            *[gr.update(value=None)
            for i in range(3)],
            app_state_dict
        )
        if user:

            # Load profile picture + preferences
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)
            pref_updates = featureviz.refresh_preferences(user)

            return (
                *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
                if i == 5 else gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
                for i in range(7)],
                gr.update(value=pic),
                *pref_updates,
                app_state_dict
            )
        return show_login(status, user)
    
    def show_settings(status, user, app_state_dict):
        """SETTINGS TAB"""
        return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
            if i == 6 else gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
            for i in range(7)],
            app_state_dict
        )
    
    def redirect():
        """Redirects to the login after logging out/deleting account"""
        return (
            *[gr.update(elem_classes="animate__animated animate__fadeInLeft")
            if i == 0 else gr.update(elem_classes="hidden-tab") 
            for i in range(7)],
            None  
        )
        
    # SIDEBAR NAVIGATION BUTTONS
    with gr.Sidebar():
        with gr.Column():
            # Logo
            gr.Markdown(
                f"""
                <div>
                    <img src='/gradio_api/file=static/MyCNN.jpg' 
                        style='width: 300px; height: auto; display: block; margin-left: auto; margin-right: auto;' 
                        alt='MyCNN Logo' />
                </div>
                """
            )
            
            # Navigation buttons
            login_button = gr.Button("Login")
            dashboard_button = gr.Button("Dashboard")
            train_button = gr.Button("Train")
            test_button = gr.Button("Test")
            gradcam_button = gr.Button("GradCAM")
            featureviz_button = gr.Button("FeatureViz")
            settings_button = gr.Button("Settings", variant="secondary")
            
            # Profile picture in sidebar
            global_profile_pic = gr.Image(label=None, show_label=False, interactive=False, buttons=[], elem_id = "profile-pic")

            # Toggle assistant visibility
            toggle_assistant = gr.Button("Toggle Assistant", variant="outline")

            # Button callbacks
            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, app_state]
            )

            # Login
            login.login_btn.click(
                fn=login.login_pipeline,
                inputs=[login.login_username, login.login_password],
                outputs=[login.login_status, login.current_user]
            )

            # Dashboard button
            dashboard_button.click(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components(), app_state]
            )

            # Train button
            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, train.lr_input, train.epoch_input, train.bs_input, train.dropout_input, train.archtype_input, train.layer1_input, train.layer2_input, train.layer3_input, train.layer4_input, app_state]
            )

            # Test button
            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, test.bs_input, app_state]
            )

            # GradCAM button
            gradcam_button.click(
                fn=show_gradcam,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, app_state]
            )
            
            # FeatureViz button
            featureviz_button.click(
                fn=show_featureviz,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, featureviz.layer_name, featureviz.channel_idx_input, app_state]
            )

            # Settings button
            settings_button.click(
                fn=show_settings,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, app_state]
            )

            # Delete Account button
            settings.delete_acc_btn.click(
                fn=redirect,
                inputs=[],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, login.current_user]
            )

            # Logout button
            settings.logout_btn.click(
                fn=redirect,
                inputs=[],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, login.current_user]
            )

            # Close app button
            settings.close_btn.click(
                fn=settings.close_app,
                inputs=[],
                outputs=[]
            )

            # Toggle assistant visibility
            toggle_assistant.click(
                fn=lambda visible: (
                    gr.update(elem_classes = "hidden" if visible else ""),
                    not visible
                ),
                inputs=[assistant_state],
                outputs=[chat_panel, assistant_state]
            )

# LAUNCH APP
if __name__ == "__main__":
    demo.launch(css=css_path, theme=gr.themes.Citrus(), footer_links=[], allowed_paths=["."])