import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub


# 准备数据
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

train_dir = "../10_food_classes_10_percent/train/"
test_dir = "../10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                          target_size=IMAGE_SHAPE,
                                                          batch_size=BATCH_SIZE,
                                                          class_mode="categorical")

test_data = train_datagen.flow_from_directory(test_dir,
                                              target_size=IMAGE_SHAPE,
                                              batch_size=BATCH_SIZE,
                                              class_mode="categorical")

resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"


# 建立模型
def create_model(url, num_classes=10):
    feature_extractor_layer = hub.KerasLayer(url,
                                             trainable=False,  # freeze the underlying patterns
                                             name='feature_extraction_layer',
                                             input_shape=IMAGE_SHAPE + (3,))  # define the input image shape

    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,  # use the feature extraction layer as the base
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')  # create our own output layer
    ])

    return model


resnet_model = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)

# EfficientNet模型
# efficientnet_model = create_model(model_url=efficientnet_url, # use EfficientNetB0 TensorFlow Hub URL
#                                   num_classes=train_data_10_percent.num_classes)

resnet_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])

# resnet_history = resnet_model.fit(train_data_10_percent,
#                                   epochs=5,
#                                   steps_per_epoch=len(train_data_10_percent),
#                                   validation_data=test_data,
#                                   validation_steps=len(test_data))

print(resnet_model.summary())

