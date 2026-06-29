import gradio as gr
import os
import json
from gtts import gTTS
import io
from groq import Groq, GroqError
import httpx
import requests

# Gets API key from env
api_key = os.getenv("api_key")

# Sets API key to Groq client
client = Groq(api_key=api_key)

class MyCNN_Assistant():
    """Handles the UI and functionality within the MyCNN assistant"""
    def __init__(self):
        # No initialisation needed, but class structure kept for clarity
        pass

    def build_ui(self):
        """Builds the UI chat panel of the MyCNN assistant. Returns
         the chatpanel, accordion, chatbot component, message component, audio component."""
        with gr.Column(scale=4, elem_id="chat-panel") as chat_panel:
                with gr.Accordion("MyCNN Assistant") as acc:
                    global_chatbot = gr.Chatbot(elem_id="global-chat")
                    msg = gr.Textbox(label="Your message", elem_id="global-chat-input")
                    audio_out = gr.Audio(autoplay=True, elem_id="assistant-audio")
        return chat_panel, acc, global_chatbot, msg, audio_out

    def clean_history(self, history):
        """Cleans the chat history by stripping out extra metadata and keeping only
        role + content pairs"""
        return [{"role": m["role"], "content": m["content"]} for m in history]
    
    def speak(self, text):
         """Convert assistant text into audio using gTTS"""
         from system.backend_config.config import SOUNDSENABLED
         # Only generate audio if sound is enabled in config
         if SOUNDSENABLED:
            tts = gTTS(text, tld="co.in", lang="en")
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
         return None
    
    def respond(self, message, chat_history, app_state):
        """Main response handler for the assistant."""
        # Ensure chat history exists
        try:
            chat_history = chat_history or []

            # Add user message to history
            chat_history.append({"role": "user", "content": message})

            # If user is not logged in, give a generic message, block access, use TTS
            if app_state["user"] is None:
                assistant_msg = "Please log in to use the assistant."
                chat_history.append({"role": "assistant", "content": assistant_msg})
                audio = self.speak(assistant_msg)
                return "", audio, chat_history

            # Build system context for the model
            system_context = {
                "current_tab": app_state["current_tab"],
                "user": app_state["user"],
                "last_error": app_state["last_error"],
                "model_status": app_state["model_status"],
                "dashboard": app_state["dashboard_info"]
            }

            # Description of all UI components (used as grounding for the LLM)
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
            
            # System prompt for the LLM
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

            # Clean history for LLM consumption
            safe_history = self.clean_history(chat_history)

            # Combine system prompt + conversation
            messages = [system_prompt] + safe_history

            # Query Groq model
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages
            )

            # Extract assistant message
            assistant_msg = response.choices[0].message.content

            # Add assistnat message to history
            chat_history.append({"role": "assistant", "content": assistant_msg})

            # Generate audio output
            audio = self.speak(assistant_msg)

            # Return empty text (UI uses chat history instead), audio, and updated history
            return "", audio, chat_history
        # Exception if an API connection error occurs
        except GroqError as e:
            return gr.Warning(f"Groq API error. Please check your API key or internet connection: {str(e)}")
        
        except httpx.ConnectError:
            return gr.Warning("Unable to reach Groq servers. Check your internet connection.")
        
        except httpx.HTTPError as e:
            return gr.Warning(f"HTTP error while contacting Groq {str(e)}")
        
        except Exception as e:
            return gr.Warning(f"Unexpected error: {str(e)}")