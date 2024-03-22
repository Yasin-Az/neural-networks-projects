import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def predict_and_visualize(model, class_indices, dataset_path, num_images=9):
    """
    Predict and visualize the model's performance on a random sample of images from the dataset.

    Args:
        model (tf.keras.Model): The pre-trained model to be used for prediction.
        class_indices (dict): A dictionary mapping class names to their corresponding indices.
        dataset_path (str): The path to the dataset directory.
        num_images (int, optional): The number of random images to visualize. Default is 9.

    Returns:
        None
    """

    def load_and_preprocess_image(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    categories = os.listdir(dataset_path)
    chosen_images = []

    for category in random.sample(categories, num_images):
        category_path = os.path.join(dataset_path, category)
        image_name = random.choice(os.listdir(category_path))
        image_path = os.path.join(category_path, image_name)
        chosen_images.append((image_path, class_indices[category]))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for (image_path, true_label), ax in zip(chosen_images, axes.flatten()):
        img_array = load_and_preprocess_image(image_path)
        prediction = model.predict(img_array)
        predicted_label_index = np.argmax(prediction)
        predicted_label = [key for key, value in class_indices.items() if value == predicted_label_index][0]

        ax.imshow(tf.keras.preprocessing.image.load_img(image_path))
        ax.axis('off')

        if predicted_label_index == true_label:
            ax.set_title('Correct Prediction: {}'.format(predicted_label), color='green')
        else:
            ax.set_title('Incorrect Prediction: {}'.format(predicted_label), color='red')

    plt.tight_layout()
    plt.show()