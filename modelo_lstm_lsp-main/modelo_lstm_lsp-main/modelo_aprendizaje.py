import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Directorios de datos
train_dir = r'C:\Users\cseba\OneDrive\Escritorio\Proyecto_final\modelo_lstm_lsp-main\gestos\train'
val_dir = r'C:\Users\cseba\OneDrive\Escritorio\Proyecto_final\modelo_lstm_lsp-main\gestos\val'

# Parámetros de entrenamiento
img_size = (224, 224)
batch_size = 32
num_epochs = 50  # Aumentado para permitir un entrenamiento más exhaustivo
fine_tune_at = 100  # Número de capas a descongelar para el fine-tuning
initial_learning_rate = 0.0001

# Preparación de los datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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

# Añadir nuevas capas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Aumentar Dropout para mayor regularización
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Entrenamiento del modelo
model.fit(
    train_generator, 
    epochs=num_epochs, 
    validation_data=val_generator, 
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Guardar el mejor modelo
model.save('modelo_gestos_mejorado_final.keras')
