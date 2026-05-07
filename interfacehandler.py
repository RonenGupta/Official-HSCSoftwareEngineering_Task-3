import gradio as gr
from modelhandler import ModelManager
from securityhandler import SecurityManager
from graphhandler import GraphManager
from torchvision import transforms

class Train_Tab():
    def __init__(self):
        gr.Markdown("# Training Demo")

        with gr.Tab("Train"):
            transform_options = [
                "Resize(256, 256)",
                "CenterCrop(224)",
                "RandomResizedCrop(224)",
                "RandomHorizontalFlip",
                "ToTensor",
                "Normalize"
            ]

            self.lr_input= gr.Slider(0.0001, 0.1, label="Learning Rate")
            self.epoch_input = gr.Number(label="Epochs")
            self.bs_input = gr.Number(label="Batch Size")
            self.train_path_input = gr.Textbox(label="Training Folder Path", placeholder="/absolute/path/to/your/dataset")
            self.train_transforms_input = gr.CheckboxGroup(choices=transform_options, label="Select transforms for training!")
            self.layer4_input = gr.Checkbox(label="Use Layer4")
            self.train_btn = gr.Button("Start Training")
            with gr.Row(equal_height=True):
                self.train_status = gr.Textbox(label="Status")
                self.train_graph = gr.Plot(label="Loss Curve")

            self.train_btn.click(
            fn=self.train_pipeline,
            inputs=[self.train_path_input, self.epoch_input, self.lr_input, self.bs_input, self.layer4_input, self.train_transforms_input],
            outputs=[self.train_status, self.train_graph])
        
    def train_pipeline(self, train_folder, epochs, lr, bs, layer4, selected_transforms):
        
        path = train_folder.name if hasattr(train_folder, "name") else train_folder

        final_transforms = []

        if "Resize(256, 256)" in selected_transforms:
            final_transforms.append(transforms.Resize((224, 224)))
        
        if "CenterCrop(224)" in selected_transforms:
            final_transforms.append(transforms.CenterCrop(224))
        
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
            return "Invalid training folder", None
        
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
    
    


