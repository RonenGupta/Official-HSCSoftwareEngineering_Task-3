import gradio as gr
from modelhandler1 import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager

with gr.Blocks() as demo:
    gr.Markdown("# Training Demo")

    with gr.Tab("Train"):
        transform_options = [
            "Resize(256, 256)",
            "RandomResizedCrop(224)",
            "RandomHorizontalFlip",
            "ToTensor",
            "Normalize"
        ]

        lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate")
        epoch_input = gr.Number(label="Epochs")
        bs_input = gr.Number(label="Batch Size")
        train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")
        train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")
        layer4_input = gr.Checkbox(label="Use Layer4")
        train_btn = gr.Button("Start Training")
        train_status = gr.Textbox(label="Status")
        train_graph = gr.Plot(label="Loss Curve")
        
        def train_pipeline(train_folder, epochs, lr, bs, layer4):
            
            path = train_folder.name if hasattr(train_folder, "name") else train_folder

            
            if not SecurityManager(path).validate_path():
                return "Invalid training folder"
            
            mm = ModelManager()
            mm.train_transforms_dataset(None, path, bs)
            mm.build(layer4)

            
            gen =  mm.train(epochs, lr)
            try:
                while True:
                    log = next(gen)
                    yield log, None
            except StopIteration as e:
                losses = e.value
                gm = GraphManager(losses, epochs)
                fig = gm.update_loss()
            yield log, fig
        
        train_btn.click(
            fn=train_pipeline,
            inputs=[train_path_input, epoch_input, lr_input, bs_input, layer4_input],
            outputs=[train_status, train_graph]
)

if __name__ == "__main__":
    demo.launch()

