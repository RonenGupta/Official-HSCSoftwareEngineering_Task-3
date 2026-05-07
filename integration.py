from interfacehandler import Train_Tab
import gradio as gr


with gr.Blocks() as demo:
    tt = Train_Tab()

if __name__ == "__main__":
    demo.launch()