from interfacehandler import Train_Tab, Test_Tab, LoginSignUp
import gradio as gr

with gr.Blocks(theme=gr.themes.Monochrome(), fill_height=True, fill_width=True) as demo:

    with gr.Group():
        with gr.Group(visible=True) as login_tab:
            login = LoginSignUp()
        with gr.Group(visible=False) as train_tab:
            train = Train_Tab(login.current_user)
        with gr.Group(visible=False) as test_tab:
            test = Test_Tab(login.current_user)

    def show_login(status, user):
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False)
            )

    def show_train(status, user):
        if user:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
    )
    
    def show_test(status, user):
        if user:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True)
            )
        return (
             gr.update(visible=True),
             gr.update(visible=False),
             gr.update(visible=False)
        )
            
    
    login.login_btn.click(
        fn=show_login,
        inputs=[login.login_status, login.current_user],
        outputs=[login_tab, train_tab, test_tab]
    )
    with gr.Sidebar():
        gr.Markdown("Navigation - MyCNN.")
        with gr.Column():
            login_button = gr.Button("Login Tab")
            train_button = gr.Button("Train Tab")
            test_button = gr.Button("Test Tab")

            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab]

            )
            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab]
            )

            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user],
                outputs=[login_tab, train_tab, test_tab]
            )

if __name__ == "__main__":
    demo.launch()