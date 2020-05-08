import tensorflow as tf
import math
import cifar10_dataset.data_loader as ld
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# Training parameters
batch_size = 100
num_classes = 10
validation_size = 5000


def wide_basic(inputs, in_planes, out_planes, stride):
    if stride != 1 or in_planes != out_planes:
        skip_c = tf.keras.layers.Conv2D(out_planes, kernel_size=1, strides=stride, use_bias=True,
                                        padding='same')(inputs)
    else:
        skip_c = inputs

    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, trainable=True)(inputs)  # Original
    # implementation had decay. Changed for momentum.
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=1, use_bias=True, padding='same')(x)
    x = tf.keras.layers.Dropout(rate=0.1, trainable=True)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True,
                                           trainable=True)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, use_bias=True, padding='same')(x)

    x = tf.add(skip_c, x)

    return x


def wide_layer(out, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    i = 0
    for strid in strides:
        out = wide_basic(out, in_planes, out_planes, strid)
        in_planes = out_planes
        i += 1

    return out


def make_resnet_filter(inputs, depth=28, widen_factor=10):
    n = (depth - 4) / 6
    k = widen_factor
    print('| Wide-Resnet %dx%d' % (depth, k))
    nstages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nstages[0], kernel_size=3, strides=1, use_bias=True, padding='same')(inputs)
    x = wide_layer(x, nstages[0], nstages[1], n, stride=1)
    x = wide_layer(x, nstages[1], nstages[2], n, stride=2)
    x = wide_layer(x, nstages[2], nstages[3], n, stride=2)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, trainable=True)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.AvgPool2D([8, 8])(x)
    x = tf.reshape(x, (-1, 640))
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 90:
        optim_factor = 3
    elif epoch > 60:
        optim_factor = 2
    elif epoch > 30:
        optim_factor = 1

    return init * math.pow(0.2, optim_factor)


@tf.function
def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model, loss_object, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def main():
    # Data loading
    (x_train, y_train), (x_val, y_val), (x_test, y_test), _ = ld.get_train_val_test_data(validation_size)

    # Set TF random seed to improve reproducibility
    tf.random.set_seed(1234)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    input_shape = x_train.shape[1:]
    labels_shape = y_train.shape[1]
    inputs = tf.keras.Input(shape=input_shape)

    wide_resnet = make_resnet_filter(inputs, depth=28, widen_factor=10)

    lr_tf = learning_rate(0.1, 0)  # Initial learning rate
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_tf, momentum=0.9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    epochs = 120
    for epoch in tqdm(range(1, epochs)):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        optimizer.lr.assign(learning_rate(0.1, epoch))
        batched_train_data = train_data.batch(batch_size)
        # Perform training
        for batch_x, batch_y in batched_train_data:
            train_step(batch_x, batch_y, wide_resnet, loss_object, optimizer, train_loss, train_accuracy)
        # Perform validation
        batched_val_data = validation_data.batch(batch_size)
        for batch_x, batch_y in batched_val_data:
            test_step(batch_x, batch_y, wide_resnet, loss_object, test_loss, test_accuracy)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


if __name__ == "__main__":
    main()
