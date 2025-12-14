import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

IMAGE_DIR = 'archive (3)/'
INPUT_SIZE = 160
NUM_CLASSES = 2
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

USE_PRETRAINED = True
FEATURE_EXTRACT = True

from sklearn.utils import class_weight


def load_images_from_folder(image_dir, input_size=INPUT_SIZE):
    dataset = []
    labels = []
    for label_name, label_idx in (('no', 0), ('yes', 1)):
        folder = os.path.join(image_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for image_name in os.listdir(folder):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, image_name)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((input_size, input_size))
            dataset.append(np.array(img))
            labels.append(label_idx)
    return np.array(dataset), np.array(labels)


print('Loading images...')
dataset, label = load_images_from_folder(IMAGE_DIR, INPUT_SIZE)

if len(dataset) == 0:
    raise SystemExit('No images found. Check the archive (3)/ directory structure (should contain no/ and yes/).')

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0, stratify=label)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

normalization_layer = tf.keras.layers.Normalization(axis=-1)
normalization_layer.adapt(x_train)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.08),
])


def build_model(input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))

    x = normalization_layer(inputs)

    x = data_augmentation(x)

    if USE_PRETRAINED:
        base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(input_size, input_size, 3), pooling='avg')
        base.trainable = not FEATURE_EXTRACT
        x = base(x, training=not FEATURE_EXTRACT)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
    else:
        x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()

def train_and_save(model, x_train, y_train, x_val, y_val):
    checkpoint_path = 'best_model.keras'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    ]

    cw_vals = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(cw_vals[i]) for i in range(len(cw_vals))}
    print('Class weights:', class_weights)

    if USE_PRETRAINED and FEATURE_EXTRACT:
        print('Stage 1: training top classifier (base frozen)')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history1 = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=INITIAL_EPOCHS, validation_data=(x_val, y_val), shuffle=True, callbacks=callbacks, class_weight=class_weights)

        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break
        if base_model is not None:
            print('Unfreezing last layers of base model for fine-tuning')
            fine_tune_at = max(0, len(base_model.layers) - 30)
            for i, l in enumerate(base_model.layers):
                l.trainable = i >= fine_tune_at
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history2 = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS, initial_epoch=INITIAL_EPOCHS, validation_data=(x_val, y_val), shuffle=True, callbacks=callbacks, class_weight=class_weights)
        else:
            print('Base model not found — skipping fine-tune stage')
    else:

        lr = 1e-4 if (USE_PRETRAINED and not FEATURE_EXTRACT) else 1e-3
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS, validation_data=(x_val, y_val), shuffle=True, callbacks=callbacks, class_weight=class_weights)

    try:
        best = tf.keras.models.load_model('best_model.keras')
        best.save('BrainTumor10EpochsCategorical.h5')
        print('Saved best model to BrainTumor10EpochsCategorical.h5')
    except Exception as e:
        print('Warning: could not convert checkpoint to h5:', e)

        model.save('BrainTumor10EpochsCategorical.h5')

    eval_model = locals().get('best', model)

    history_obj = None
    if 'history2' in locals():
        history_obj = locals().get('history2')
    elif 'history1' in locals():
        history_obj = locals().get('history1')
    elif 'history' in locals():
        history_obj = locals().get('history')

    print("\nEvaluating model...")
    try:
        y_pred_probs = eval_model.predict(x_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
    except Exception as e:
        print('Could not run prediction on validation set:', e)
        y_pred = np.zeros_like(y_val)

    report = classification_report(y_val, y_pred, target_names=['no', 'yes'], output_dict=True)
    print("\nPerformance Report")
    print(classification_report(y_val, y_pred, target_names=['no', 'yes']))

    plt.figure(figsize=(6, 2))
    plt.axis('off')
    table_data = [[k, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1-score']:.2f}"]
                  for k, v in report.items() if k in ['no', 'yes']]
    plt.table(cellText=table_data, colLabels=["Class", "Precision", "Recall", "F1-score"], loc='center')
    plt.title("Model Performance Table")
    plt.show()

    if history_obj is not None and hasattr(history_obj, 'history'):
        h = history_obj.history
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        if 'accuracy' in h:
            plt.plot(h.get('accuracy', []), label='Train Acc')
        if 'val_accuracy' in h:
            plt.plot(h.get('val_accuracy', []), label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1,2,2)
        if 'loss' in h:
            plt.plot(h.get('loss', []), label='Train Loss')
        if 'val_loss' in h:
            plt.plot(h.get('val_loss', []), label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        print('No training history available to plot accuracy/loss curves.')

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nGenerating Grad-CAM...")
    last_conv = None
    try:
        if hasattr(eval_model, 'get_layer'):
            try:
                last_conv = 'mobilenetv2_1.00_160'
                _ = eval_model.get_layer(last_conv)
            except Exception:
                for layer in reversed(eval_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv = layer.name
                        break
    except Exception:
        last_conv = None

    if last_conv is None:
        print('Could not find a Conv2D layer for Grad-CAM. Skipping Grad-CAM visualization.')
    else:
        try:
            grad_model = tf.keras.models.Model(inputs=eval_model.inputs, outputs=[eval_model.get_layer(last_conv).output, eval_model.output])
            idx = np.random.randint(0, len(x_val))
            sample = np.expand_dims(x_val[idx], axis=0)
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(sample)
                class_idx = np.argmax(predictions[0])
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()[0]
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            img = x_val[idx]
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_uint = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap_color, 0.4, 0)

            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.title(f'Original Image (True: {y_val[idx]}, Pred: {class_idx})')
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(superimposed)
            plt.title('Grad-CAM Visualization')
            plt.axis('off')
            plt.show()
        except Exception as e:
            print('Grad-CAM generation failed:', e)



if __name__ == '__main__':
    train_and_save(model, x_train, y_train, x_test, y_test)