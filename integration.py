from interfacehandler import Dashboard, Train_Tab, Test_Tab, LoginSignUp, GradCAM, FeatureViz, Settings
import gradio as gr
from pathlib import Path
import json

USER_DB = 'users.json'

css = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

.spaced-row {
    gap: 24px !important;
}

body, .gradio-container {
    background: transparent !important;   
    min-height: 100vh !important;   
    padding: 20px !important;
}

.gradio-container {
    position: relative; !important;
}

.gradio-container,
div[class*="gr-"],
div[class*="svelte-"],
.tabs,
.tab-nav,
.tabitem,
.form,
.gap,
.block,
fieldset,
.g-row,
.g-col,
div.row,
div.column {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border-color: #ffffff !important;
}

div.row:has(> .gr-markdown),
.block p,
div.row p {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.models-section {
    margin-top: 32px !important; 
    margin-bottom: 32px !important;
}

.gr-markdown {
    margin-top: 20px !important;
    margin-bottom: 20px !important;
}

h1, h2, h3, .gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #ff9800 !important; 
    font-weight: 700 !important;
}

.block, 
fieldset, 
.accordion, 
details, 
input, 
textarea, 
select {
    border: 1px solid #e2e8f0 !important; 
    border-radius: 12px !important; 
}

.tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 12px !important;
}

.fixed-width-container, 
.fixed-width-container .gradio-audio,
.fixed-width-container audio {
    max-width: 450px !important; 
    width: 100% !important;
    overflow-x: hidden !important;
}

.hidden-tab {
    display: none !important;
}

#profile-pic {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    position: absolute !important;
    top: 20px !important;
    right: 20px !important;
    z-index: 9999 !important;
}

#profile-pic img {
    border-radius: 50%; !important;
    object-fit: cover; !important;

}
"""
gr.set_static_paths(paths=[Path.cwd() / "static"])

with gr.Blocks(fill_height=True, fill_width=True) as demo:
    global_profile_pic = gr.Image(label=None, show_label=False, interactive=False, buttons=[], elem_id = "profile-pic")

    with gr.Group():
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
            
    def show_login(status, user):
            return (
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            )
    
    def show_dashboard(status, user):
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
                *card_updates
            )
        return show_login(status, user)

    def show_train(status, user):
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),

            )
        return show_login(status, user)

    
    def show_test(status, user):
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
            )
        return show_login(status, user)
    
    def show_gradcam(status, user):
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
            )
        return show_login(status, user)
    
    def show_featureviz(status, user):
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
            )
        return show_login(status, user)
    
    def show_settings(status, user):
        return (
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="animate__animated animate__fadeInLeft")
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
        gr.Markdown(
            f"""
            <div>
                <img src='/gradio_api/file=static/MyCNN.jpg' 
                     style='width: 300px; height: auto; display: block; margin-left: auto; margin-right: auto;' 
                     alt='MyCNN Logo' />
            </div>
            """
        )
        with gr.Column():
            login_button = gr.Button("Login")
            dashboard_button = gr.Button("Dashboard")
            train_button = gr.Button("Train")
            test_button = gr.Button("Test")
            gradcam_button = gr.Button("GradCAM")
            featureviz_button = gr.Button("FeatureViz")
            settings_button = gr.Button("Settings", variant="secondary")

            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab]
            )

            login.login_btn.click(
                fn=login.login_pipeline,
                inputs=[login.login_username, login.login_password],
                outputs=[login.login_status, login.current_user]
            ).then(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components()]
            )

            dashboard_button.click(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components()]
            )

            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic]
            )

            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic]
            )

            gradcam_button.click(
                fn=show_gradcam,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic]
            )

            featureviz_button.click(
                fn=show_featureviz,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic]
            )

            settings_button.click(
                fn=show_settings,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab]
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
    demo.launch(css=css, theme=gr.themes.Citrus(), footer_links=[], allowed_paths=["."])