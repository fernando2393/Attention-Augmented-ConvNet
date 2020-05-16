import sys
sys.path.append("..")
import tensorflow as tf
import cifar10_dataset.data_loader as ld
import models.widenet28_10 as widenet
from engine.training.custom_training import TrainingEngine
from engine.learning_rate.wide_learning_rate import WideLearningRate
from tensorflow.keras.optimizers import SGD
from os import path

# Training parameters
batch_size = 128
validation_size = 5000
augmented = True  # Set to true to implement augmented conv2D layers


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)
    # Data loading
    x_train, y_train, x_val, y_val, x_test, y_test, _ = ld.get_train_val_test_datasets(validation_size)
    # Set TF random seed to improve reproducibility
    tf.random.set_seed(1234)
    # Obtain training, validation and test data
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # Obtain the input placeholder of the model
    input_shape = x_train.shape[1:]
    inputs = tf.keras.Input(shape=input_shape)
    # Define the model architecture
    model = widenet.make_resnet_filter(inputs, depth=28, widen_factor=10, augmented=augmented)
    print(model.summary())
    if path.isfile("widenet_weights.ckpt"):
        model.load_weights("widenet_weights.ckpt")
    training_module = TrainingEngine(model)
    training_module.lr_scheduler = WideLearningRate
    training_module.optimizer = SGD(lr=training_module.lr_scheduler.get_learning_rate(epoch=0),
                                    momentum=0.9, nesterov=True)
    # Train de model
    training_module.fit(train_data,
                        validation_data,
                        batch_size=batch_size,
                        epochs=60,
                        data_augmentation=True)
    model.save_weights("widenet_weights.ckpt")
    # Perform evaluation over test data
    scores = training_module.evaluate(test_data)
    print('Test loss:', scores[1])
    print('Test accuracy:', scores[0] * 100)
    model.save_weights("./widenet_selfatt.ckpt")


if __name__ == "__main__":
    main()
