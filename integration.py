from interfacehandler import Train_Tab, Test_Tab
import gradio as gr


with gr.Blocks() as demo:
    train = Train_Tab()
    test = Test_Tab()

if __name__ == "__main__":
    demo.launch()