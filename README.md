# Tikki

This repository contains Mökki Tikki Android application with an AI opponent trained using reinforcement learning. It has two main components:

- `AI/`: Python scripts for training the reinforcement learning agent.
- `Android/`: Android Studio project for the Android application.

## Setup Instructions

### AI Training
1. Navigate to the `AI/` directory:
   ```bash
   cd AI
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python Agent_V1-0_PPO.py
   ```

### Android App
1. Open the `Android/` directory in Android Studio.
2. Sync the project with Gradle files.
3. Build and run the app on an Android device or emulator.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
