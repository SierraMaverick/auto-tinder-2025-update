import tensorflow as tf
tf.compat.v1.disable_eager_execution() # Ensure TF1 compatibility mode for the session
import person_detector
from PIL import Image # person_detector.get_person returns a PIL Image object

print("Attempting to load person detection graph...")
try:
    detection_graph = person_detector.open_graph()
    print("Person detection graph loaded successfully.")

    print("\nAttempting to detect a person in an image...")
    test_image_path = "images/unclassified/test_4tz3kjldfj3482.jpg"

    # get_person requires a session
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:
            pil_image = person_detector.get_person(test_image_path, sess)
            if pil_image:
                print(f"Person detection successful for {test_image_path}. Result is a PIL Image: {type(pil_image)}")
                # You could save or show the image here if in an environment that supports it
                # pil_image.save("detected_person.jpg")
            else:
                print(f"No person detected or an issue occurred for {test_image_path}.")

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}. This might be an issue with graph path or the test image path.")
except Exception as e:
    print(f"An error occurred during the person_detector smoke test: {e}")
    import traceback
    traceback.print_exc()

print("\nPerson detector smoke test finished.")
