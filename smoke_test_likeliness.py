import tensorflow as tf
tf.compat.v1.disable_eager_execution() # Ensure TF1 compatibility mode for the session
from likeliness_classifier import Classifier
import numpy as np # Classifier uses np.squeeze

print("Attempting to instantiate Classifier...")
try:
    classifier = Classifier(graph="./tf/training_output/retrained_graph.pb",
                            labels="./tf/training_output/retrained_labels.txt")
    print("Classifier instantiated successfully.")

    print("\nAttempting to classify an image...")
    # Create a dummy JPG file for testing if a real one is problematic
    # For now, let's assume the path is correct and accessible by the script
    test_image_path = "images/unclassified/test_4tz3kjldfj3482.jpg"
    
    # The classify method expects a string path to the image
    predictions = classifier.classify(test_image_path)
    print(f"Classification result for {test_image_path}: {predictions}")
    
    classifier.close()
    print("Classifier closed.")

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}. This might be an issue with graph/label paths or the test image path.")
except Exception as e:
    print(f"An error occurred during the likeliness_classifier smoke test: {e}")
    import traceback
    traceback.print_exc()

print("\nLikeliness classifier smoke test finished.")
