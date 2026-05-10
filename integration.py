from interfacehandler import Train_Tab, Test_Tab, LoginSignUp
import gradio as gr

with gr.Blocks() as demo:

    login = LoginSignUp()
    train = Train_Tab()   
    test = Test_Tab()     

if __name__ == "__main__":
    demo.launch()