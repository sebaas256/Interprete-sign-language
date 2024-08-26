import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
<<<<<<< HEAD
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
=======
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
>>>>>>> 62dfa55bc7961f9eafaf2ca8faa6ed053b56c3bf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Directorios de datos
<<<<<<< HEAD
train_dir = r'C:\Users\lilia\OneDrive\Escritorio\Proyecto_final\modelo_lstm_lsp-main\gestos\train'
val_dir = r'C:\Users\lilia\OneDrive\Escritorio\Proyecto_final\modelo_lstm_lsp-main\gestos\val'

# Parámetros de entrenamiento
img_size = (224, 224)
batch_size = 32
num_epochs = 20  # Aumentar el número de épocas para un mejor ajuste
fine_tune_at = 100  # Número de capas a descongelar para el fine-tuning

# Preparación de los datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='categorical'
)

# Cargar el modelo pre-entrenado
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = True

# Descongelar las últimas capas para el fine-tuning
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
=======
train_dir = r'C:\Users\cseba\OneDrive\Escritorio\Proyecto_final\modelo_lstm_lsp-main\gestos\train'
val_dir = r'C:\Users\cseba\OneDrive\Escritorio\Proyecto_final\modelo_lstm_lsp-main\gestos\val'

# Parámetros de entrenamiento_gene
img_size = (224, 224)
batch_size = 32
num_epochs = 10

# Preparación de los datos
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Cargar el modelo pre-entrenado
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False
>>>>>>> 62dfa55bc7961f9eafaf2ca8faa6ed053b56c3bf

# Añadir nuevas capas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
<<<<<<< HEAD
x = Dropout(0.3)(x)  # Añadir Dropout para evitar sobreajuste
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
=======
>>>>>>> 62dfa55bc7961f9eafaf2ca8faa6ed053b56c3bf
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_generator, epochs=num_epochs, validation_data=val_generator)

# Guardar el modelo
<<<<<<< HEAD
model.save('modelo_gestos_mejorado.h5')
=======
model.save('modelo_gestos.keras')
>>>>>>> 62dfa55bc7961f9eafaf2ca8faa6ed053b56c3bf
