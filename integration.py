from interfacehandler import Train_Tab, Test_Tab, LoginSignUp
import gradio as gr

with gr.Blocks() as demo:

    login = LoginSignUp()

    with gr.Tab("Train") as train_tab:
        train = Train_Tab(login.current_user)

    with gr.Tab("Test") as test_tab:
        test = Test_Tab(login.current_user)

    train_tab.visible = False
    test_tab.visible = False     

    def visible_tabs(status, user):
        visible = bool(user)
        return gr.update(visible=visible), gr.update(visible=visible)
    
    login.login_btn.click(
        fn=visible_tabs,
        inputs=[login.login_status, login.current_user],
        outputs=[train_tab, test_tab]
    )

if __name__ == "__main__":
    demo.launch()