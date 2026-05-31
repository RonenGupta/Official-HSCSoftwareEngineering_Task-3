from interfacehandler import Dashboard, Train_Tab, Test_Tab, LoginSignUp, GradCAM, Settings
import gradio as gr

css = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

.spaced-row {
    gap: 24px !important;
}

body, .gradio-container {
    background: #d3d3d3 !important;   
    min-height: 100vh !important;   
    padding: 20px !important;
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
"""

with gr.Blocks(fill_height=True, fill_width=True) as demo:

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
            
    def show_login(status, user):
            return (
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
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

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                welcome,
                count,
                last_model,
                last_acc,
                last_time,

                *card_updates
            )
        return show_login(status, user)

    def show_train(status, user):
        if user:
            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
            )
        return show_login(status, user)

    
    def show_test(status, user):
        if user:
            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            )
        return show_login(status, user)
    
    def show_gradcam(status, user):
        if user:
            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
            )
        return show_login(status, user)
    
    def show_settings():
        return (
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
            None  
        )
        
    with gr.Sidebar():
        gr.Markdown("Navigation - MyCNN.")
        with gr.Column():
            login_button = gr.Button("Login")
            dashboard_button = gr.Button("Dashboard")
            train_button = gr.Button("Train")
            test_button = gr.Button("Test")
            gradcam_button = gr.Button("GradCAM")
            settings_button = gr.Button("Settings", variant="secondary")

            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            login.login_btn.click(
                fn=login.login_pipeline,
                inputs=[login.login_username, login.login_password],
                outputs=[login.login_status, login.current_user]
            ).then(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, *dashboard.get_card_components()]
            )

            dashboard_button.click(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, *dashboard.get_card_components()]
            )

            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            gradcam_button.click(
                fn=show_gradcam,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            settings_button.click(
                fn=show_settings,
                inputs=[],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            settings.logout_btn.click(
                fn=logout,
                inputs=[],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, settings_tab, login.current_user]
            )

            settings.close_btn.click(
                fn=settings.close_app,
                inputs=[],
                outputs=[]
            )

if __name__ == "__main__":
    demo.launch(css=css, theme=gr.themes.Citrus(), footer_links=["settings"])