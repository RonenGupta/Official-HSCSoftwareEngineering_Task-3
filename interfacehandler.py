import gradio as gr
from modelhandler1 import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager
from torchvision import transforms

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
        
        def train_pipeline(train_folder, epochs, lr, bs, layer4, selected_transforms):
            
            path = train_folder.name if hasattr(train_folder, "name") else train_folder

            final_transforms = []

            if "Resize(256, 256)" in selected_transforms:
                final_transforms.append(transforms.Resize((224, 224)))
            
            if "RandomResizedCrop(224)" in selected_transforms:
                final_transforms.append(transforms.RandomResizedCrop(224, scale=(0.6, 1.0)))
            
            if "RandomHorizontalFlip" in selected_transforms:
                final_transforms.append(transforms.RandomHorizontalFlip(0.5))

            if "ToTensor" in selected_transforms:
                final_transforms.append(transforms.ToTensor())
            
            if "Normalize" in selected_transforms:
                final_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]))
                
            if not final_transforms:
                train_transforms = None
            else:
                train_transforms = transforms.Compose(final_transforms)

            if not SecurityManager(path).validate_path():
                return "Invalid training folder"
            
            mm = ModelManager()
            mm.train_transforms_dataset(train_transforms, path, bs)
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
            inputs=[train_path_input, epoch_input, lr_input, bs_input, layer4_input, train_transforms_input],
            outputs=[train_status, train_graph]
)

if __name__ == "__main__":
    demo.launch()

