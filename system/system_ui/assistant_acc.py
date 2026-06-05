import gradio as gr
import os
import json
from groq import Groq

api_key = os.getenv("api_key")

client = Groq(api_key=api_key)

class MyCNN_Assistant():
    def __init__(self):
        pass

    def build_ui(self):
        with gr.Column(scale=4, elem_id="chat-panel"):
                with gr.Accordion("MyCNN Assistant"):
                    global_chatbot = gr.Chatbot(elem_id="global-chat")
                    msg = gr.Textbox(label="Your message", elem_id="global-chat-input")
        return global_chatbot, msg
    
    def clean_history(self, history):
        return [{"role": m["role"], "content": m["content"]} for m in history]
    
    def respond(self, message, chat_history, app_state):
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})

        if app_state["user"] is None:
            assistant_msg = "Please log in to use the assistant."
            chat_history.append({"role": "assistant", "content": assistant_msg})
            return "", chat_history

        system_context = {
            "current_tab": app_state["current_tab"],
            "user": app_state["user"],
            "last_error": app_state["last_error"],
            "model_status": app_state["model_status"],
            "dashboard": app_state["dashboard_info"]
        }

        app_components = """
        Available Tabs and Components:

        1. Dashboard:
        - Shows welcome message, number of saved models, last trained model, last accuracy, and last training time.
        - Displays model cards (up to 10) with details: model name, accuracy, loss, epochs, date, architecture.
        - Each card has: Notes field + Save Notes button, Loss Curve plot, Accuracy Curve plot, Confusion Matrix plot, Download, Delete, and Generate PDF buttons.

        2. Train Tab:
        - Training Folder Path input.
        - Hyperparameters: Learning Rate slider, Epochs, Batch Size, Dropout Rate slider.
        - Early Stopping: checkbox + Patience slider.
        - Transforms: Checkbox group (Resize, CenterCrop, RandomResizedCrop, flips, rotation, ColorJitter, etc.).
        - Architecture: ResNet18/34/50/101/152 dropdown + Layer1 to Layer4 checkboxes.
        - Train button, training status box, Loss Curve plot, Accuracy Curve plot.
        - System Analytics: GPU usage plot, CPU/RAM plot, live analytics JSON.
        - Save Model section with model name input and Save button.

        3. Test Tab:
        - Saved Model dropdown selector.
        - Batch Size input.
        - Testing Folder Path input.
        - Refresh Saved Models button.
        - Test button, test status, Confusion Matrix plot.

        4. GradCAM Tab:
        - Image Folder Path input.
        - Model selector dropdown.
        - Transforms checkbox group.
        - Batch Size input.
        - Refresh Saved Models button.
        - View Augmentation Examples button + output image.
        - Generate GradCAM button with: Original Image, GradCAM visualization, Predicted Class label.

        5. FeatureViz Tab:
        - Model selector dropdown.
        - Visualization Mode: "Channel Visualization" or "Activation Maps".
        - Layer name input, Channel Index number.
        - Input Image upload (for Activation Maps).
        - Image Size, Optimization Steps, Learning Rate sliders.
        - Generate Feature Visualization button + output image.
        """

        system_prompt = {
            "role": "system", 
            "content": f"""You are MyCNN Assistant, a helpful guide for the MyCNN application.

        Rules:
        - Use the Current App State and App components below as your main source of information.
        - Do not answer questions which are unrelated to MyCNN, or CNN's in general.
        - You can describe the known tabs and components in the app: Dashboard, Train, Test, GradCAM, FeatureViz, Settings, and Login.
        - Answer based on what makes sense for these components even if not all details are in the current state or app components.
        - Do NOT invent or hallucinate new features, buttons, models, or capabilities that are not part of the actual application.
        - If you don't have specific data (like last trained model), base your answer on general knowledge of the app.
        - Keep responses short, clear, and natural. Maximum 2 sentences. Do not format your response with bold, indents, or any other technique. Just plain text.
        Current App State:
        {json.dumps(system_context, indent=2)}
        App Components: {app_components}"""             
        
        }

        safe_history = self.clean_history(chat_history)

        messages = [system_prompt] + safe_history

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )

        assistant_msg = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": assistant_msg})

        return "", chat_history