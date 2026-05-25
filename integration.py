from interfacehandler import Train_Tab, Test_Tab, LoginSignUp, GradCAM, Settings
import gradio as gr

css = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

.spaced-row {
    gap: 24px !important;
}

.gradio-container {
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
        with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as train_tab:
            train = Train_Tab(login.current_user)
        with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as test_tab:
            test = Test_Tab(login.current_user)
        with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as gradcam_tab:
            gradcam = GradCAM(login.current_user)
        with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as settings_tab:
            settings = Settings()
            
    def show_login(status, user):
            return (
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            )

    def show_train(status, user):
        if user:
            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
            )
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
        )

    
    def show_test(status, user):
        if user:
            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            )
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
    )
    
    def show_gradcam(status, user):
        if user:
            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
            )
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft")
        )
    
    def show_settings():
        return (
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="animate__animated animate__fadeInLeft")
        )
        
    with gr.Sidebar():
        gr.Markdown("Navigation - MyCNN.")
        with gr.Column():
            login_button = gr.Button("Login Tab")
            train_button = gr.Button("Train Tab")
            test_button = gr.Button("Test Tab")
            gradcam_button = gr.Button("GradCAM Tab")
            settings_button = gr.Button("Settings Tab")

            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab, gradcam_tab, settings_tab]

            )
            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            gradcam_button.click(
                fn=show_gradcam,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

            settings_button.click(
                fn=show_settings,
                inputs=[],
                outputs=[login_tab, train_tab, test_tab, gradcam_tab, settings_tab]
            )

if __name__ == "__main__":
    demo.launch(css=css, theme=gr.themes.Citrus())