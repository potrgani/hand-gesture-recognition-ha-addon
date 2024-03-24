# Hand Gesture Recognition HA-Addon

The Hand Gesture Recognition Add-on is a Home Assistant add-on that enables the recognition of hand gestures using MediaPipe and publishes them to MQTT. This add-on allows users to integrate hand gesture recognition into their Home Assistant setup for various automation and control tasks.

The model can recognize seven classes (i.e. 👍, 👎, ✌️, ☝️, ✊, 👋, 🤟).

# Features

- Real-time hand gesture recognition using MediaPipe
- Integration with MQTT for communication with Home Assistant
- Customizable MQTT configuration options
- Supports various architectures including aarch64, amd64, armhf, armv7, and i386

# Installation

- Add the repository to your Home Assistant instance.
   
  [![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https://github.com/potrgani/hand-gesture-recognition-ha-addon)
  
- Install the Hand Gesture Recognition Add-on.
- Installation will take a while, so grab a cup of coffee ☕️ and relax.
- Configure the add-on with your MQTT broker details and RTSP URL.
- Start the add-on.

# Configuration

The add-on provides the following configuration options:

- **RTSP URL:** RTSP URL of the video stream.
- **MQTT Host:** Hostname or IP address of the MQTT broker.
- **MQTT Port:** Port number of the MQTT broker (default: 1883).
- **MQTT Username:** Username for MQTT authentication.
- **MQTT Password:** Password for MQTT authentication.
- **MQTT Topic:** MQTT topic to publish hand gesture status (default: "hand_gesture_status").

# Usage

1. Once the add-on is started, it will continuously capture video input and perform hand gesture recognition.
2. Recognized hand gestures will be published to the specified MQTT topic.
3. Users can integrate these gesture recognition results into Home Assistant automations and scripts.

# Notes

- Ensure that your Home Assistant instance is configured to use MQTT and that the MQTT broker is reachable from the add-on.
- For optimal performance, it is recommended to run this add-on on a device with sufficient hardware resources.

