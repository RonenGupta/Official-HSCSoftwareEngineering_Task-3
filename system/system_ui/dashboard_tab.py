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

# Initialize pygame mixer for sound notifications
pygame.mixer.init()

# Load notification sound
music_path = os.path.join(MUSIC_FOLDER, "ping.mp3")
pygame.mixer.music.load(music_path)

# Instantiate managers for models, graphs, user profiles
mm = ModelManager()
gm= GraphManager()
pm = ProfileManager()

class Dashboard():
    """Handles the UI layout and functionality of the Dashboard tab"""
    def __init__(self, current_user):
        # Store the currently logged-in user
        self.current_user = current_user

        # Display the MyCNN banner/logo at the top of the dashboard
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

        # Welcome message section
        with gr.Group():
            self.welcome = gr.Markdown(
                                    """
                                    <div style='padding: 15px; border-radius: 10px; background: #ffffff; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);'>
                                        <h3>Welcome!</h3>
                                        <p>Loading user data...</p>
                                    </div>
                                    """
                                )
        
        # Display summary statistics: model count, last model trained, last accuracy, last time trained
        with gr.Row():
            with gr.Column():
                self.model_count = gr.Textbox(label="Saved Models", interactive=False)
            with gr.Column():
                self.last_model = gr.Textbox(label = "Last Trained Model", interactive=False)
            with gr.Column():
                self.last_accuracy = gr.Textbox(label="Last Accuracy", interactive=False)
            with gr.Column():
                self.last_time = gr.Textbox(label="Last Time Trained", interactive=False)
        
        # Section for displaying up to 10 model cards
        with gr.Group(elem_classes = "models-section"):
            gr.Markdown("### Your Models")

            # Container for model cards
            with gr.Column() as self.model_cards:  
                self.card_slots = []
                cards_per_row = 2

                # Create 10 card slots (5 rows x 2 columns)
                for i in range(0, 10, cards_per_row):
                    with gr.Row(equal_height=True):
                        for j in range(cards_per_row):
                            if i+j >= 10:
                                break
                            
                            # Each card is an accordion with model details
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

                                    # Store all components for this card
                                    self.card_slots.append((acc, slot_model_name, slot_md, slot_notes, save_notes_btn, slot_loss, slot_acc, slot_cm, download_btn, delete_btn, pdf_btn))

        # Attach button callbacks for each card slot
        for slot in self.card_slots:
            (acc, slot_model_name, slot_md, slot_notes, save_notes_btn, slot_loss, slot_acc, slot_cm, download_btn, delete_btn, pdf_btn) = slot

            # Save notes button
            save_notes_btn.click(
                fn=self.update_notes,
                inputs=[self.current_user, slot_model_name, slot_notes], 
                outputs=[]
            )
            
            # Download model button
            download_btn.click(
                fn=self.download_user_models,
                inputs=[self.current_user, slot_model_name],
                outputs=[]
            )

            # Delete model button
            delete_btn.click(
                fn=self.delete_model,
                inputs=[self.current_user, slot_model_name],
                outputs=[]
            )
            
            # Generate PDF report button
            pdf_btn.click(
                fn=self.generate_pdf,
                inputs=[self.current_user, slot_model_name]
            )

    def load_dashboard(self, username):
        """Load dashboard summary information for the given user."""
        try:
            # If no user logged in, return placeholder values
            if not username:
                return "<h3>Not logged in</h3>", 0, "-", "-"
            
            # Load user database
            with open(USER_DB, "r") as f:
                users = json.load(f)

            # Retrieve this user's saved models
            models = users[username]["models"]
            count = len(models)

            # If user has no models, return "friendly" message
            if count == 0:
                return (
                    f"<h3>Welcome {username}!</h3><p>Noooooo noooooo train a model!! :p</p>",
                    0,
                    "No models yet :(",
                    "N/A",
                    "N/A",
                    {}
                )
            
            # Identify the most recently saved model
            last_model = list(models.keys())[-1]
            last_acc = models[last_model].get("accuracy", "Unknown")
            last_time = models[last_model].get("date", "Unknown")

            # Return dashboard summary
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
            # If something fails, return safe defaults
            return gr.Warning(str(e)), 0, "-", "-", "-", {}
    
    def build_model_cards(self, models):
        """Build the UI updates for all model cards. Takes dict of models and returns updates"""
        updates = []

        # Hide all card slots and reset their values
        for _ in self.card_slots:
            updates.extend([
                # Ensure model name and model metadata text do not have any value
                gr.update(value="", visible=False) if i in (1, 2) else gr.update(visible=False)
                for i in range(11)
            ])
        
        # Fill card slots with actual model data
        for i, (model_name, data) in enumerate(models.items()):
            if i >= len(self.card_slots):
                break # Only show up to 10 cards
            
            # Build metadata text block
            html = (
            f"Model: {model_name}\n"
            f"Accuracy: {data.get('accuracy', 'Unknown')}\n"
            f"Loss: {data.get('loss', 'Unknown')}\n"
            f"Epochs: {data.get('epochs', 'Unknown')}\n"
            f"Date: {data.get('date', 'Unknown')}\n"
            f"Architecture: {data.get('architecture')}\n"
        )

            # Each card uses 11 update slots
            base = i * 11

            updates[base] = gr.update(visible=True)

            updates[base+1] = gr.update(value=model_name, visible=False)
            
            updates[base+2] = gr.update(value=html, visible=True)

            updates[base+3] = gr.update(value=data.get("notes", "No notes"), visible=True)

            updates[base+4] = gr.update(visible=True)

            # Loss curve
            if data.get("loss_curve"):
                loss_fig = gm.update_loss(data["loss_curve"], len(data["loss_curve"]))
                updates[base + 5] = gr.update(value=loss_fig, visible=True)

            # Accuracy curve
            if data.get("accuracy_curve"):
                acc_fig = gm.update_accuracy(data["accuracy_curve"], len(data["accuracy_curve"]))
                updates[base + 6] = gr.update(value=acc_fig, visible=True)

            # Confusion matrix
            if data.get("confusion_matrix"):
                labels, preds = data["confusion_matrix"]
                class_names = data.get("class_names", [])
                cm_fig = gm.update_confusion_matrix(labels, preds, class_names)
                updates[base + 7] = gr.update(value=cm_fig, visible=True)

            # Buttons
            updates[base+8] = gr.update(visible=True)
            updates[base+9] = gr.update(visible=True)
            updates[base+10] = gr.update(visible=True)

        # Finally return all updates
        return updates
    
    def get_card_components(self):
        """Flatten all card slot components into a single list"""
        return [c for slot in self.card_slots for c in slot]
    
    def delete_model(self, username, model_name):
        """Delete a saved model and all associated files (weights, PDF, temp images)."""
        try:
            # Load user database
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            # Ensure model exists
            if model_name not in users[username]["models"]:
                return gr.Warning("Model not found.")
            
            # Delete model weights
            model_path = f"saved_models/{username}_{model_name}.pth"
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Delete PDF report
            pdf_path = f"reports/{username}_{model_name}.pdf"
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
            # Delete temporary images used in PDF generation
            tmp_dir = "reports/tmp"
            if os.path.exists(tmp_dir):
                for file in os.listdir(tmp_dir):
                    if file.startswith(f"{username}_{model_name}") or file.startswith(f"loss_{username}_{model_name}") or file.startswith(f"accuracy_{username}_{model_name}") or file.startswith(f"cm_{username}_{model_name}"):
                        os.remove(os.path.join(tmp_dir, file))

            # Remove model entry from database
            del users[username]["models"][model_name]

            # Save updated DB
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
            
            # Play sound + notification
            if SOUNDSENABLED:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play()
            if NOTIFICATIONS_ENABLED:
                return gr.Info("Model deleted!", duration=6)
        
        # Error handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e))

    def download_user_models(self, username, model):
        """Trigger a model download for the user."""
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
        """Generate a PDF report for a saved model, including a loss curve,
        accuracy curve, confusion matrix, model metadata table"""
        try:
            # Load user DB
            with open(USER_DB) as f:
                users = json.load(f)
            
            data = users[username]["models"][model_name]

            # Ensure model has been tested
            if data.get("confusion_matrix") is None or data.get("class_names") is None:
                return gr.Warning("You must test this model before generating a PDF report.")

            # Prepare directories
            os.makedirs("reports", exist_ok=True)
            path = f"reports/{username}_{model_name}.pdf"
            tmp_path = f"reports/tmp"
            os.makedirs(tmp_path, exist_ok=True)

            # Create PDF document
            doc = SimpleDocTemplate(path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Add logo
            logo_path = "static/MyCNN.jpg"
            if os.path.exists(logo_path):
                story.append(Image(logo_path, width=240, height=150))
                story.append(Spacer(1, 20))
            
            # Title + subtitle
            title = f"<para align='center'><font size=24><b>{model_name} - Model Report </b></font></para>"
            story.append(Paragraph(title, styles["Title"]))
            story.append(Spacer(1, 12))
            subtitle = f"<para align='center'><font size=24><b>Made by Ronen - MyCNN</b></font></para>"
            story.append(Paragraph(subtitle))
            story.append(Spacer(1, 30))

            # Summary header
            story.append(Paragraph("<b>Model Summary </b>", styles["Heading2"]))
            story.append(Spacer(1, 12))

            # Loss curve
            if "loss_curve" in data and "epochs" in data:
                fig = gm.update_loss(data["loss_curve"], data["epochs"])
                loss_path = f"{tmp_path}/loss_{username}_{model_name}.png"
                fig.savefig(loss_path)
                story.append(Paragraph("<b>Loss Curve</b>", styles["Heading2"]))
                story.append(Image(loss_path, width=300, height=220))
                story.append(Spacer(1, 20))
            
            # Accuracy curve
            if "accuracy_curve" in data and "epochs" in data:
                fig = gm.update_accuracy(data["accuracy_curve"], data["epochs"])
                acc_path = f"{tmp_path}/accuracy_{username}_{model_name}.png"
                fig.savefig(acc_path)
                story.append(Paragraph("<b>Accuracy Curve</b>", styles["Heading2"]))
                story.append(Image(acc_path, width=300, height=220))
                story.append(Spacer(1, 20))
            
            # Confusion matrix
            if "confusion_matrix" in data and "class_names" in data:
                stored = data["confusion_matrix"]
                fig = gm.rebuild_confusion_matrix(stored, data["class_names"])
                cm_path = f"{tmp_path}/cm_{username}_{model_name}.png"
                fig.savefig(cm_path)
                story.append(Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]))
                story.append(Image(cm_path, width=300, height=300))
                story.append(Spacer(1, 20))

            # Metadata table
            table_data = [["Field", "Value"]]
            for k, v in data.items():
                table_data.append([k, str(v)])
            
            # Table styling
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

            # Append table to document, add a space below
            story.append(table)
            story.append(Spacer(1, 20))

            # Footer below
            footer = Paragraph(
                "<para align='center'><font size=10 color='grey'>Generated by MyCNN</font></para>",
                styles["Normal"]
            )
            story.append(Spacer(1, 40))
            story.append(footer)

            # Build PDF
            doc.build(story)

            # Play sound + notification
            if SOUNDSENABLED:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play()
            if NOTIFICATIONS_ENABLED:
                return gr.Info("PDF generated!", duration=6)
        
        # Error handling in case of Exception
        except Exception as e:
            return gr.Warning(f"Failed to generate PDF: {e}")
    
    def update_notes(self, username, model_name, new_notes):
        """Update the notes field for a specific saved model"""
        try:
            # Load DB
            with open(USER_DB, "r") as f:
                users = json.load(f)
            
            # Ensure model exists
            if model_name not in users.get(username, {}).get("models", {}):
                return gr.Warning(f"Model '{model_name}' not found.")

            # Update notes
            users[username]["models"][model_name]["notes"] = new_notes

            # Save DB
            with open(USER_DB, "w") as f:
                json.dump(users, f, indent=4)
            
            # Play sound + notification
            if SOUNDSENABLED:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play()

            if NOTIFICATIONS_ENABLED:
                return gr.Info("Notes updated!", duration=6)
        
        # Error handling in case of Exception
        except Exception as e:
            return gr.Warning(str(e))