import argparse
import sys
import time
import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import logging
import json
json_file_path = '/data/options.json'
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
with open(json_file_path, 'r') as file:
    json_data = file.read()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logging level to INFO or any other level you prefer

# Define a handler to output logs to standard output
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set the level for this handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parse the JSON data
data = json.loads(json_data)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MQTT configuration
mqtt_broker_address = data.get("mqtt_host")
mqtt_port = data.get("mqtt_port")
mqtt_topic = data.get("mqtt_topic")
mqtt_username = data.get("mqtt_username")
mqtt_password = data.get("mqtt_password")

# MQTT client
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(mqtt_username, mqtt_password)



# Function to restart MQTT connection
def restart_mqtt_connection():
    mqtt_client.disconnect()
    time.sleep(1)  # Wait for the client to disconnect
    mqtt_client.connect(mqtt_broker_address, mqtt_port, 60)
    print("Restarted MQTT connection")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
    else:
        print("Connection to MQTT Broker failed with code", rc)
        restart_mqtt_connection()


mqtt_client.on_connect = on_connect
# Timer for restarting MQTT connection every 3 minutes
mqtt_restart_interval = 180  # 3 minutes (in seconds)
mqtt_last_restart_time = time.time()

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
FRAME_COUNT = 0  # Counter for saving images

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
  global FRAME_COUNT, mqtt_last_restart_time # Declare FRAME_COUNT as a global variable
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the gesture recognition model bundle.
      num_hands: Max number of hands can be detected by the recognizer.
      min_hand_detection_confidence: The minimum confidence score for hand
        detection to be considered successful.
      min_hand_presence_confidence: The minimum confidence score of hand
        presence score in the hand landmark detection.
      min_tracking_confidence: The minimum confidence score for the hand
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

  # Start capturing video input from the camera
  #cap = cv2.VideoCapture(camera_id)
  #rtsp_url = "rtsp://192.168.100.120:8080/h264.sdp"
  cap = cv2.VideoCapture(data.get("rtsp_url"))
 # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
 # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
 # cv2.namedWindow('gesture_recognition', cv2.WINDOW_NORMAL)
 # cv2.resizeWindow('gesture_recognition', 640, 480)
  
 

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 0)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Label box parameters
  label_text_color = (255, 255, 255)  # white
  label_font_size = 1
  label_thickness = 2

  recognition_frame = None
  recognition_result_list = []
  

  def save_result(result: vision.GestureRecognizerResult,
                  unused_output_image: mp.Image, timestamp_ms: int):
      global FPS, COUNTER, START_TIME

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      recognition_result_list.append(result)
      COUNTER += 1

  # Initialize the gesture recognizer model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.GestureRecognizerOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          num_hands=num_hands,
                                          min_hand_detection_confidence=min_hand_detection_confidence,
                                          min_hand_presence_confidence=min_hand_presence_confidence,
                                          min_tracking_confidence=min_tracking_confidence,
                                          result_callback=save_result)
  recognizer = vision.GestureRecognizer.create_from_options(options)

    
  prev_handedness_value = None

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

            # Increment the frame count
    FRAME_COUNT += 1

        # Check if it's time to restart MQTT connection
    current_time = time.time()
    if current_time - mqtt_last_restart_time >= mqtt_restart_interval:
        restart_mqtt_connection()
        mqtt_last_restart_time = current_time

        # Save an image every 1 second
    if FRAME_COUNT % 25 == 0:
            cv2.imwrite(f'frame.jpg', image)
            #print(f'Image saved: frame_{FRAME_COUNT}.jpg')

            # Load the saved image for gesture recognition
            saved_image = cv2.imread(f'frame.jpg')

            # Perform gesture recognition on the saved image
            rgb_saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
            mp_saved_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_saved_image)

            recognizer.recognize_async(mp_saved_image, time.time_ns() // 1_000_000)


    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if recognition_result_list:
      #print(recognition_result_list)
      # Draw landmarks and write the text for each hand.
      for hand_index, hand_landmarks in enumerate(
          recognition_result_list[0].hand_landmarks):
        # Calculate the bounding box of the hand
        x_min = min([landmark.x for landmark in hand_landmarks])
        y_min = min([landmark.y for landmark in hand_landmarks])
        y_max = max([landmark.y for landmark in hand_landmarks])

        # Convert normalized coordinates to pixel values
        frame_height, frame_width = current_frame.shape[:2]
        x_min_px = int(x_min * frame_width)
        y_min_px = int(y_min * frame_height)
        y_max_px = int(y_max * frame_height)

        #Get hand 
        if recognition_result_list[0].handedness:
           handedness_info = recognition_result_list[0].handedness[0]
           handedness_value = handedness_info[0].display_name
           #print(handedness_value)

        # Get gesture classification results
        if recognition_result_list[0].gestures:
          
          gesture = recognition_result_list[0].gestures[hand_index]
          category_name = gesture[0].category_name
          score = round(gesture[0].score, 2)
          result_text = f'{category_name} ({score})'
        
        

          # Compute text size
          text_size = \
          cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                          label_thickness)[0]
          text_width, text_height = text_size

          # Calculate text position (above the hand)
          text_x = x_min_px
          text_y = y_min_px - 10  # Adjust this value as needed

          # Make sure the text is within the frame boundaries
          if text_y < 0:
            text_y = y_max_px + text_height

          # Draw the text
          cv2.putText(current_frame, result_text, (text_x, text_y),
                      cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                      label_text_color, label_thickness, cv2.LINE_AA)
          
          #print(result_text, (text_x, text_y))
        #hand_status = (handedness_value+' '+category_name) #right/left + category name
        hand_status = category_name
        #print (hand_status+str(score))
         # Check if the handedness status has changed
        if hand_status != prev_handedness_value and score > 0.6:
              on_connect()
              mqtt_client.publish(mqtt_topic, hand_status)
              logger.info(hand_status)
              prev_handedness_value = hand_status
              print (hand_status)

        # Draw hand landmarks on the frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in
          hand_landmarks
        ])
        mp_drawing.draw_landmarks(
          current_frame,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

      recognition_frame = current_frame
      recognition_result_list.clear()

    #if recognition_frame is not None:
       # cv2.imshow('gesture_recognition', recognition_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of gesture recognition model.',
      required=False,
      default='gesture_recognizer.task')
  parser.add_argument(
      '--numHands',
      help='Max number of hands that can be detected by the recognizer.',
      required=False,
      default=1)
  parser.add_argument(
      '--minHandDetectionConfidence',
      help='The minimum confidence score for hand detection to be considered '
           'successful.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minHandPresenceConfidence',
      help='The minimum confidence score of hand presence score in the hand '
           'landmark detection.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minTrackingConfidence',
      help='The minimum confidence score for the hand tracking to be '
           'considered successful.',
      required=False,
      default=0.5)
  # Finding the camera ID can be very reliant on platform-dependent methods.
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.numHands), args.minHandDetectionConfidence,
      args.minHandPresenceConfidence, args.minTrackingConfidence,
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
