# HSC Software Engineering Task 3

## Overview

This repository contains an interactive CNN-based application built with Gradio for training, testing, visualizing, and managing image classification models. The app supports user authentication, a dashboard, model training, testing with confusion matrices, GradCAM explainability, feature visualization, and an assistant panel.

## Key Features

- User login and signup with secure password handling
- Gradio web UI with multiple tabs: Login, Dashboard, Train, Test, GradCAM, FeatureViz, Settings
- CNN model training pipeline for image datasets
- Saved model management and evaluation
- Test mode with confusion matrix visualization
- GradCAM activation maps for model explainability
- Feature visualization for selected CNN layers and channels
- Profile preferences and settings
- Built-in assistant sidebar for guidance and interaction

## Repository Structure

- `main.py` - Launches the Gradio application and wires together the UI tabs
- `system/` - Core application logic and UI tab implementations
  - `system/system_ui/` - Gradio tab classes and interface components
  - `system/system_functions/` - Backend functionality for models, profiling, security, and graph handling
  - `system/backend_config/` - Configuration settings
- `datasets/hymenoptera_data/` - Example dataset structure with `train` and `test` splits for `ants` and `bees`
- `saved_models/` - Persisted trained model files
- `users.json` - User database file with saved user profiles and preferences
- `static/` - CSS, audio, and profile image assets
- `tests/` - Pytest unit tests

## Requirements

Install dependencies from `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a .env file and insert a Groq API key from this link: `https://console.groq.com/keys`.

```bash
api_key = "YOUR_API_KEY"
```

## Running the App

Start the application with:

```bash
python main.py
```

Then open the local Gradio URL shown in the terminal.

## Usage

1. Open the app in your browser.
2. Create a new account or log in using existing credentials.
3. Use the Dashboard to view saved models and user activity.
4. Train new models from the Train tab using your dataset.
5. Evaluate saved models in the Test tab and view confusion matrices.
6. Explore model explainability with GradCAM and FeatureViz.
7. Adjust profile settings and use the assistant sidebar.

## Dataset

The repository includes a sample dataset structure under `datasets/hymenoptera_data/`:

- `train/ants`
- `train/bees`
- `test/ants`
- `test/bees`

To use this dataset in any dataset entry, simply pass in the base folder, and the program will interpret train and test folders.

## Notes (IMPORTANT)

- `users.json` stores user credentials and profile preferences.
- Saved model files are located in `saved_models/`.
- The app uses `pygame` for audio notifications and `gradio` for the web UI.
- If you are uploading a dataset, it MUST be in the following structure (Can have more classes):

- `datasets/hymenoptera_data/` - Example Dataset
  - `datasets/hymenoptera_data/train` - Training directory of the dataset
    - `datasets/hymenoptera_data/train/ants` - Class1
    - `datasets/hymenoptera_data/train/bees` - Class2
  - `datasets/hymenoptera_data/test` - Testing directory of the dataset
    - `datasets/hymenoptera_data/test/ants` - Class1
    - `datasets/hymenoptera_data/test/bees` - Class2

## License

This repository is released under the license included in `LICENSE`.
