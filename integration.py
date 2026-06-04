from interfacehandler import Dashboard, Train_Tab, Test_Tab, LoginSignUp, GradCAM, FeatureViz, Settings
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import os
import json
from groq import Groq

USER_DB = 'users.json'
load_dotenv()
api_key = os.getenv("api_key")

client = Groq(api_key=api_key)

css = """
@import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

.spaced-row {
    gap: 24px !important;
}

body, .gradio-container {
    background: transparent !important;   
    min-height: 100vh !important;   
    padding: 20px !important;
}

.gradio-container {
    position: relative; !important;
}

.gradio-container,
div[class*="gr-"],
div[class*="svelte-"],
.tabs,
.tab-nav,
.tabitem,
.form,
.gap,
.block,
fieldset,
.g-row,
.g-col,
div.row,
div.column {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border-color: #ffffff !important;
}

div.row:has(> .gr-markdown),
.block p,
div.row p {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.models-section {
    margin-top: 32px !important; 
    margin-bottom: 32px !important;
}

.gr-markdown {
    margin-top: 20px !important;
    margin-bottom: 20px !important;
}

h1, h2, h3, .gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #ff9800 !important; 
    font-weight: 700 !important;
}

.block, 
fieldset, 
.accordion, 
details, 
input, 
textarea, 
select {
    border: 1px solid #e2e8f0 !important; 
    border-radius: 12px !important; 
}

.tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 12px !important;
}

.fixed-width-container, 
.fixed-width-container .gradio-audio,
.fixed-width-container audio {
    max-width: 450px !important; 
    width: 100% !important;
    overflow-x: hidden !important;
}

.hidden-tab {
    display: none !important;
}

@keyframes flip {
    0% {
    transform: rotateY(0deg);
    }
    100% {
    transform: rotateY(180deg);
    }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
    transform: translateY(0);
    animation-timing-function: cubic-bezier(0.215 0.610, 0.355, 1.00);
    }

    40%, 43% {
    transform: translateY(-30px);
    animation-timing-function: cubic-bezier(0.755 0.050, 0.855, 0.060);
    }

    70% {
    transform: translateY(-15px);
    animation-timing-function: cubic-bezier(0.755 0.050, 0.855, 0.060);
    }

    90% {
    transform: translateY(-4px);
   }
}

#profile-pic {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    position: relative;
    top: 0;
    right: 0;
    z-index: 1;
    animation: flip 2s cubic-bezier(0.65, 0, 0.35, 1) infinite alternate ;
    transform-style: preserve-3d;
}

#profile-pic img {
    border-radius: 50%; !important;
    object-fit: cover; !important;
}

#chat-panel {
    top: 0;
    right: 0;
    width: 350px;
    height: 100vh;
    background: white;
    border-left: 2px solid #e2e8f0;
    padding: 10px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
}

#global-chat {
    flex: 1;
    overflow-y: auto;
}

#global-chat-input {
    margin-top: 10px;
}
"""

gr.set_static_paths(paths=[Path.cwd() / "static"])

with gr.Blocks(fill_height=True, fill_width=True) as demo:

    app_state = gr.State({
    "current_tab": "login",
    "user": None,
    "last_error": None,
    "model_status": None,
    "dashboard_info": {}
    })

    with gr.Row():
        with gr.Column(scale=8):
            with gr.Group(elem_classes="animate__animated animate__fadeInLeft") as login_tab:
                login = LoginSignUp()
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as dashboard_tab:
                dashboard = Dashboard(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as train_tab:
                train = Train_Tab(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as test_tab:
                test = Test_Tab(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as gradcam_tab:
                gradcam = GradCAM(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as settings_tab:
                settings = Settings(login.current_user)
            with gr.Group(elem_classes="hidden-tab animate__animated animate__fadeInLeft") as featureviz_tab:
                featureviz = FeatureViz(login.current_user)
        
        
        with gr.Column(scale=4, elem_id="chat-panel"):
            chat_visible = gr.State(value=True)
            with gr.Group(visible=True) as chat_group:
                with gr.Accordion("MyCNN Assistant"):
                    global_chatbot = gr.Chatbot(elem_id="global-chat")
                    msg = gr.Textbox(label="Your message", elem_id="global-chat-input")

            def clean_history(history):
                cleaned = []
                for msg in history:
                    cleaned.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                return cleaned
            
            def respond(message, chat_history, app_state):
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

                safe_history = clean_history(chat_history)

                messages = [system_prompt] + safe_history

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages
                )

                assistant_msg = response.choices[0].message.content

                chat_history.append({"role": "assistant", "content": assistant_msg})

                return "", chat_history
    
            msg.submit(
                respond, 
                [msg, global_chatbot, app_state], 
                [msg, global_chatbot]
                )
            
    def show_login(status, user, app_state_dict):
        app_state_dict["current_tab"] = "login"
        app_state_dict["user"] = user
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            app_state_dict
        )

    def show_dashboard(status, user, app_state_dict):
        app_state_dict["current_tab"] = "dashboard"
        app_state_dict["user"] = user

        if not user:
            gr.Info("Please enter a valid username", duration=6)

            return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),   
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"),  
            "", "", "", "", "",                    
            gr.update(value=None),                 
            *[gr.update(visible=False) for _ in dashboard.get_card_components()],
            app_state_dict
        )

        if user:
            welcome, count, last_model, last_acc, last_time, models = dashboard.load_dashboard(user)
            card_updates = dashboard.build_model_cards(models)
            with open(USER_DB, "r") as f:
                users = json.load(f)
                pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                welcome,
                count,
                last_model,
                last_acc,
                last_time,
                gr.update(value=pic),
                *card_updates,
                app_state_dict
            )
        return show_login(status, user, app_state_dict)

    def show_train(status, user, app_state_dict):
        app_state_dict["current_tab"] = "train"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                app_state_dict
            )
        return show_login(status, user)

    
    def show_test(status, user, app_state_dict):
        app_state_dict["current_tab"] = "test"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                app_state_dict
            )
        return show_login(status, user)
    
    def show_gradcam(status, user, app_state_dict):
        app_state_dict["current_tab"] = "gradcam"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                app_state_dict
            )
        return show_login(status, user)
    
    def show_featureviz(status, user, app_state_dict):
        app_state_dict["current_tab"] = "featureviz"
        app_state_dict["user"] = user
        if user:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            pic = users[user].get("preferences", {}).get("profile_picture", None)

            return (
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="animate__animated animate__fadeInLeft"),
                gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
                gr.update(value=pic),
                app_state_dict
            )
        return show_login(status, user)
    
    def show_settings(status, user, app_state_dict):
        return (
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="hidden-tab animate__animated animate__fadeInLeft"),
            gr.update(elem_classes="animate__animated animate__fadeInLeft"),
            app_state_dict
        )
    
    def logout():
        return (
            gr.update(elem_classes="animate__animated animate__fadeInLeft"), 
            gr.update(elem_classes="hidden-tab"), 
            gr.update(elem_classes="hidden-tab"),
            gr.update(elem_classes="hidden-tab"),  
            gr.update(elem_classes="hidden-tab"), 
            gr.update(elem_classes="hidden-tab"),
            gr.update(elem_classes="hidden-tab"), 
            None  
        )
        
    with gr.Sidebar():
        with gr.Column():
            gr.Markdown(
                f"""
                <div>
                    <img src='/gradio_api/file=static/MyCNN.jpg' 
                        style='width: 300px; height: auto; display: block; margin-left: auto; margin-right: auto;' 
                        alt='MyCNN Logo' />
                </div>
                """
            )

            login_button = gr.Button("Login")
            dashboard_button = gr.Button("Dashboard")
            train_button = gr.Button("Train")
            test_button = gr.Button("Test")
            gradcam_button = gr.Button("GradCAM")
            featureviz_button = gr.Button("FeatureViz")
            settings_button = gr.Button("Settings", variant="secondary")
            global_profile_pic = gr.Image(label=None, show_label=False, interactive=False, buttons=[], elem_id = "profile-pic")

            login_button.click(
                fn=show_login,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, app_state]
            )

            login.login_btn.click(
                fn=login.login_pipeline,
                inputs=[login.login_username, login.login_password],
                outputs=[login.login_status, login.current_user]
            ).then(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components(), app_state]
            )

            dashboard_button.click(
                fn=show_dashboard,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, dashboard.welcome, dashboard.model_count, dashboard.last_model, dashboard.last_accuracy, dashboard.last_time, global_profile_pic, *dashboard.get_card_components(), app_state]
            )

            train_button.click(
                fn=show_train,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, app_state]
            )

            test_button.click(
                fn=show_test,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, app_state]
            )

            gradcam_button.click(
                fn=show_gradcam,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, app_state]
            )

            featureviz_button.click(
                fn=show_featureviz,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, global_profile_pic, app_state]
            )

            settings_button.click(
                fn=show_settings,
                inputs=[login.login_status, login.current_user, app_state],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, app_state]
            )

            settings.logout_btn.click(
                fn=logout,
                inputs=[],
                outputs=[login_tab, dashboard_tab, train_tab, test_tab, gradcam_tab, featureviz_tab, settings_tab, login.current_user]
            )

            settings.close_btn.click(
                fn=settings.close_app,
                inputs=[],
                outputs=[]
            )

if __name__ == "__main__":
    demo.launch(css=css, theme=gr.themes.Citrus(), footer_links=[], allowed_paths=["."])