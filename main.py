import gradio as gr
from pathlib import Path
import json
import pygame
from system.system_ui.assistant_acc import MyCNN_Assistant
from system.system_ui.dashboard_tab import Dashboard
from system.system_ui.train_tab import Train_Tab
from system.system_ui.test_tab import Test_Tab
from system.system_ui.gradcam_tab import GradCAM
from system.system_ui.featureviz_activationmap_tab import FeatureViz
from system.system_ui.login_signup_tab import LoginSignUp
from system.system_ui.settings_tab import Settings
from system.backend_config.config import MUSIC_FOLDER

pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)

USER_DB = 'users.json'

css_path = Path("static/styles.css").read_text()

gr.set_static_paths(paths=[Path.cwd() / "static"])

with gr.Blocks(fill_height=True, fill_width=True) as demo:

    app_state = gr.State({
    "current_tab": "login",
    "user": None,
    "last_error": None,
    "model_status": None,
    "dashboard_info": {}
    })

    with gr.Row():
        with gr.Column(scale=8):
            with gr.Group(elem_classes="animate__animated animate__fadeInLeft") as login_tab:
                login = LoginSignUp()
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as dashboard_tab:
                dashboard = Dashboard(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as train_tab:
                train = Train_Tab(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as test_tab:
                test = Test_Tab(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as gradcam_tab:
                gradcam = GradCAM(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as settings_tab:
                settings = Settings(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as featureviz_tab:
                featureviz = FeatureViz(login.current_user)

        cnn_assistant = MyCNN_Assistant()
        assistant_chatbot, assistant_msg = cnn_assistant.build_ui()

        assistant_msg.submit(
            fn=cnn_assistant.respond,
            inputs=[assistant_msg, assistant_chatbot, app_state],
            outputs=[assistant_msg, assistant_chatbot]
        )

    def show_login(status, user, app_state_dict):
        app_state_dict["current_tab"] = "login"
        app_state_dict["user"] = user
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            app_state_dict
        )

    def show_dashboard(status, user, app_state_dict):
        app_state_dict["current_tab"] = "dashboard"
        app_state_dict["user"] = user

        if not user:
            gr.Info("Please enter a valid username", duration=6)

            return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),   
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            "", "", "", "", "",                    
            gr.update(value=None),                 
            *[gr.update(visible=False) for _ in dashboard.get_card_components()],
            app_state_dict
        )

        if user:
            welcome, count, last_model, last_acc, last_time, models = dashboard.load_dashboard(user)
            card_updates = dashboard.build_model_cards(models)
            with open(USER_DB, "r") as f:
                users = json.load(f)
                pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
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
        app_state_dict["current_tab"] = "train"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)
            pref_updates = train.refresh_preferences(user)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                *pref_updates,
                app_state_dict
            )
        return show_login(status, user)

    
    def show_test(status, user, app_state_dict):
        app_state_dict["current_tab"] = "test"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)
            pref_updates = test.refresh_preferences(user)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                *pref_updates,
                app_state_dict
            )
        return show_login(status, user)
    
    def show_gradcam(status, user, app_state_dict):
        app_state_dict["current_tab"] = "gradcam"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                app_state_dict
            )
        return show_login(status, user)
    
    def show_featureviz(status, user, app_state_dict):
        app_state_dict["current_tab"] = "featureviz"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)
            pref_updates = featureviz.refresh_preferences(user)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                *pref_updates,
                app_state_dict
            )
        return show_login(status, user)
    
    def show_settings(status, user, app_state_dict):
        return (
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            app_state_dict
        )
    
    def logout():
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"), 
            gr.update(elem_classes="hidden-tab"), 
            gr.update(elem_classes="hidden-tab"),
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"), 
            gr.update(elem_classes="hidden-tab"),
            gr.update(elem_classes="hidden-tab"), 
            None  
        )
        
    with gr.Sidebar():
        with gr.Column():
            gr.Markdown(
                f"""
                <div>
                    <img src='/gradio_api/file=static/MyCNN.jpg' 
                        style='width: 300px; height: auto; display: block; margin-left: auto; margin-right: auto;' 
                        alt='MyCNN Logo' />
                </div>
                """
            )

            login_button = gr.Button("Login")
            dashboard_button = gr.Button("Dashboard")
            train_button = gr.Button("Train")
            test_button = gr.Button("Test")
            gradcam_button = gr.Button("GradCAM")
            featureviz_button = gr.Button("FeatureViz")
            settings_button = gr.Button("Settings", variant="secondary")
            global_profile_pic = gr.Image(label=None, show_label=False, interactive=False, buttons=[], elem_id = "profile-pic")

            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, app_state]
            )

            login.login_btn.click(
                fn=login.login_pipeline,
                inputs=[login.login_username, login.login_password],
                outputs=[login.login_status, login.current_user]
            ).then(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components(), app_state]
            )

            dashboard_button.click(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components(), app_state]
            )

            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, train.lr_input, train.epoch_input, train.bs_input, train.dropout_input, train.archtype_input, train.layer1_input, train.layer2_input, train.layer3_input, train.layer4_input, app_state]
            )

            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, test.bs_input, app_state]
            )

            gradcam_button.click(
                fn=show_gradcam,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, app_state]
            )

            featureviz_button.click(
                fn=show_featureviz,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, featureviz.layer_name, featureviz.channel_idx_input, app_state]
            )

            settings_button.click(
                fn=show_settings,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, app_state]
            )

            settings.logout_btn.click(
                fn=logout,
                inputs=[],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, login.current_user]
            )

            settings.close_btn.click(
                fn=settings.close_app,
                inputs=[],
                outputs=[]
            )

if __name__ == "__main__":
    demo.launch(css=css_path, theme=gr.themes.Citrus(), footer_links=[], allowed_paths=["."])