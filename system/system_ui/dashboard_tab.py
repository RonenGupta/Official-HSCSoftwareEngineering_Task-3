import gradio as gr
from system.system_functions.modelhandler import ModelManager
from system.system_functions.graphhandler import GraphManager
from system.system_functions.profilehandler import ProfileManager
import os
import json
import pygame
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from system.backend_config.config import NOTIFICATIONS_ENABLED, SOUNDSENABLED, USER_DB, MUSIC_FOLDER

pygame.mixer.init()
music_path = f"/Users/RonenGupta/Desktop/HSCSoftwareEngineering_Task-3/{MUSIC_FOLDER}/ping.mp3"
pygame.mixer.music.load(music_path)

mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class Dashboard():
    def __init__(self, current_user):
        self.current_user = current_user
        gr.Markdown(
            f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <img src='/gradio_api/file=static/MyCNN.jpg' 
                     style='width: 800px; height: auto; display: block; margin-left: auto; margin-right: auto;' 
                     alt='MyCNN Logo' />
                <p style='font-size: 1.1rem; color: #555; margin-bottom: 10px;'>
                    Your central hub for training, testing, and managing convolutional models.
                </p>
            </div>
            """
        )


        with gr.Group():
            self.welcome = gr.Markdown(
                                    """
                                    <div style='padding: 15px; border-radius: 10px; background: #ffffff; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);'>
                                        <h3>Welcome!</h3>
                                        <p>Loading user data...</p>
                                    </div>
                                    """
                                )
        
        with gr.Row():
            with gr.Column():
                self.model_count = gr.Textbox(label="Saved Models", interactive=False)
            with gr.Column():
                self.last_model = gr.Textbox(label = "Last Trained Model", interactive=False)
            with gr.Column():
                self.last_accuracy = gr.Textbox(label="Last Accuracy", interactive=False)
            with gr.Column():
                self.last_time = gr.Textbox(label="Last Time Trained", interactive=False)
        
        with gr.Group(elem_classes = "models-section"):
            gr.Markdown("### Your Models")
            with gr.Column() as self.model_cards:  
                self.card_slots = []
                cards_per_row = 2
                for i in range(0, 10, cards_per_row):
                    with gr.Row(equal_height=True):
                        for j in range(cards_per_row):
                            if i+j >= 10:
                                break
                            
                            with gr.Column():
                                with gr.Accordion(label=f"Model {i+j+1}", open=False,visible=False) as acc:
                                    slot_model_name = gr.Textbox(value="", visible=False, interactive=False)
                                    slot_md = gr.Textbox(value="", visible=False, interactive=False, lines=0, show_label=False)
                                    slot_notes = gr.Textbox(label="Notes", visible=False, interactive=True)
                                    save_notes_btn = gr.Button("Save Notes", visible=False)
                                    slot_loss = gr.Plot(visible=False)
                                    slot_acc = gr.Plot(visible=False)
                                    slot_cm = gr.Plot(visible=False)
                                    download_btn = gr.Button("Download", visible=False)
                                    delete_btn = gr.Button("Delete", visible=False)
                                    pdf_btn = gr.Button("Generate PDF", visible=False)
                                    self.card_slots.append((acc, slot_model_name, slot_md, slot_notes, save_notes_btn, slot_loss, slot_acc, slot_cm, download_btn, delete_btn, pdf_btn))

        for slot in self.card_slots:
            (acc, slot_model_name, slot_md, slot_notes, save_notes_btn, slot_loss, slot_acc, slot_cm, download_btn, delete_btn, pdf_btn) = slot

            save_notes_btn.click(
                fn=self.update_notes,
                inputs=[self.current_user, slot_model_name, slot_notes], 
                outputs=[]
            )

            download_btn.click(
                fn=self.download_user_models,
                inputs=[self.current_user, slot_model_name],
                outputs=[]
            )

            delete_btn.click(
                fn=self.delete_model,
                inputs=[self.current_user, slot_model_name],
                outputs=[]
            )

            pdf_btn.click(
                fn=self.generate_pdf,
                inputs=[self.current_user, slot_model_name]
            )

    def load_dashboard(self, username):
        try:
            if not username:
                return "<h3>Not logged in</h3>", 0, "-", "-"
            
            with open(USER_DB, "r") as f:
                users = json.load(f)

            models = users[username]["models"]

            count = len(models)

            if count == 0:
                return (
                    f"<h3>Welcome {username}!</h3><p>Noooooo noooooo train a model!! :p</p>",
                    0,
                    "No models yet :(",
                    "N/A",
                    "N/A",
                    {}
                )
            
            last_model = list(models.keys())[-1]
            last_acc = models[last_model].get("accuracy", "Unknown")
            last_time = models[last_model].get("date", "Unknown")

            return (
                f"""
                ## **Welcome {username}!**
                ### Here's your latest model activity.
                """,
                count,
                last_model,
                last_acc,
                last_time,
                models,
            )
        except Exception as e:
            return gr.Warning(str(e)), 0, "-", "-", "-", {}
    
    def build_model_cards(self, models):
        
        updates = []

        for _ in self.card_slots:
            updates.extend([
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),             
                gr.update(visible=False),              
                gr.update(visible=False),
                gr.update(visible=False),                     
            ])
        
        for i, (model_name, data) in enumerate(models.items()):
            if i >= len(self.card_slots):
                break

            html = (
            f"Model: {model_name}\n"
            f"Accuracy: {data.get('accuracy', 'Unknown')}\n"
            f"Loss: {data.get('loss', 'Unknown')}\n"
            f"Epochs: {data.get('epochs', 'Unknown')}\n"
            f"Date: {data.get('date', 'Unknown')}\n"
            f"Architecture: {data.get('architecture')}\n"
        )

            base = i * 11

            updates[base] = gr.update(visible=True)

            updates[base+1] = gr.update(value=model_name, visible=False)
            
            updates[base+2] = gr.update(value=html, visible=True)

            updates[base+3] = gr.update(value=data.get("notes", "No notes"), visible=True)

            updates[base+4] = gr.update(visible=True)

            if data.get("loss_curve"):
                loss_fig = gm.update_loss(data["loss_curve"], len(data["loss_curve"]))
                updates[base + 5] = gr.update(value=loss_fig, visible=True)

            if data.get("accuracy_curve"):
                acc_fig = gm.update_accuracy(data["accuracy_curve"], len(data["accuracy_curve"]))
                updates[base + 6] = gr.update(value=acc_fig, visible=True)

            if data.get("confusion_matrix"):
                labels, preds = data["confusion_matrix"]
                class_names = data.get("class_names", [])
                cm_fig = gm.update_confusion_matrix(labels, preds, class_names)
                updates[base + 7] = gr.update(value=cm_fig, visible=True)

            updates[base+8] = gr.update(visible=True)
            updates[base+9] = gr.update(visible=True)
            updates[base+10] = gr.update(visible=True)

        return updates
    
    def get_card_components(self):
        return [c for slot in self.card_slots for c in slot]
    
    def delete_model(self, username, model_name):
        try:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            if model_name not in users[username]["models"]:
                return gr.Warning("Model not found.")
            
            model_path = f"saved_models/{username}_{model_name}.pth"
            if os.path.exists(model_path):
                os.remove(model_path)
            
            pdf_path = f"reports/{username}_{model_name}.pdf"
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
            tmp_dir = "reports/tmp"
            if os.path.exists(tmp_dir):
                for file in os.listdir(tmp_dir):
                    if file.startswith(f"{username}_{model_name}") or file.startswith(f"loss_{username}_{model_name}") or file.startswith(f"accuracy_{username}_{model_name}") or file.startswith(f"cm_{username}_{model_name}"):
                        os.remove(os.path.join(tmp_dir, file))

            del users[username]["models"][model_name]

            
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
            if SOUNDSENABLED:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play()
            if NOTIFICATIONS_ENABLED:
                return gr.Info("Model deleted!", duration=6)
            
        
        except Exception as e:
            return gr.Warning(str(e))

    def download_user_models(self, username, model):
        try:
            mm.download_model(username, model)
            if NOTIFICATIONS_ENABLED:
                gr.Info("Downloading completed successfully!", duration=8)
            if SOUNDSENABLED:
                pygame.mixer.music.play()
            return 
        except Exception as e:
            return gr.Warning(str(e))
    
    def generate_pdf(self, username, model_name):
        try:
            with open(USER_DB) as f:
                users = json.load(f)
            
            data = users[username]["models"][model_name]

            if data.get("confusion_matrix") is None or data.get("class_names") is None:
                return gr.Warning("You must test this model before generating a PDF report.")

            os.makedirs("reports", exist_ok=True)
            path = f"reports/{username}_{model_name}.pdf"
            tmp_path = f"reports/tmp"
            os.makedirs(tmp_path, exist_ok=True)
            doc = SimpleDocTemplate(path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            logo_path = "static/MyCNN.jpg"
            if os.path.exists(logo_path):
                story.append(Image(logo_path, width=240, height=150))
                story.append(Spacer(1, 20))

            title = f"<para align='center'><font size=24><b>{model_name} - Model Report </b></font></para>"
            story.append(Paragraph(title, styles["Title"]))
            story.append(Spacer(1, 12))
            subtitle = f"<para align='center'><font size=24><b>Made by Ronen - MyCNN</b></font></para>"
            story.append(Paragraph(subtitle))
            story.append(Spacer(1, 30))

            story.append(Paragraph("<b>Model Summary </b>", styles["Heading2"]))
            story.append(Spacer(1, 12))

            if "loss_curve" in data and "epochs" in data:
                fig = gm.update_loss(data["loss_curve"], data["epochs"])
                loss_path = f"{tmp_path}/loss_{username}_{model_name}.png"
                fig.savefig(loss_path)
                story.append(Paragraph("<b>Loss Curve</b>", styles["Heading2"]))
                story.append(Image(loss_path, width=300, height=220))
                story.append(Spacer(1, 20))
            
            if "accuracy_curve" in data and "epochs" in data:
                fig = gm.update_accuracy(data["accuracy_curve"], data["epochs"])
                acc_path = f"{tmp_path}/accuracy_{username}_{model_name}.png"
                fig.savefig(acc_path)
                story.append(Paragraph("<b>Accuracy Curve</b>", styles["Heading2"]))
                story.append(Image(acc_path, width=300, height=220))
                story.append(Spacer(1, 20))
            
            if "confusion_matrix" in data and "class_names" in data:
                stored = data["confusion_matrix"]
                fig = gm.rebuild_confusion_matrix(stored, data["class_names"])
                cm_path = f"{tmp_path}/cm_{username}_{model_name}.png"
                fig.savefig(cm_path)

                story.append(Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]))
                story.append(Image(cm_path, width=300, height=300))
                story.append(Spacer(1, 20))

            table_data = [["Field", "Value"]]
            for k, v in data.items():
                table_data.append([k, str(v)])
            
            table = Table(table_data, colWidths=[150, 300])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ff9800")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))

            story.append(table)
            story.append(Spacer(1, 20))

            footer = Paragraph(
                "<para align='center'><font size=10 color='grey'>Generated by MyCNN</font></para>",
                styles["Normal"]
            )
            story.append(Spacer(1, 40))
            story.append(footer)

            doc.build(story)
            if SOUNDSENABLED:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play()
            if NOTIFICATIONS_ENABLED:
                return gr.Info("PDF generated!", duration=6)
        
        except Exception as e:
            return gr.Warning(f"Failed to generate PDF: {e}")
    
    def update_notes(self, username, model_name, new_notes):
        try:
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            if model_name not in users.get(username, {}).get("models", {}):
                return gr.Warning(f"Model '{model_name}' not found.")

            users[username]["models"][model_name]["notes"] = new_notes

            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
            if SOUNDSENABLED:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play()
            if NOTIFICATIONS_ENABLED:
                return gr.Info("Notes updated!", duration=6)
        except Exception as e:
            return gr.Warning(str(e))