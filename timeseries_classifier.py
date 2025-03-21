import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Erweiterte Funktion zum Einlesen und Segmentieren von PhyPhox-Daten
def read_phyphox_segments(file_path, segment_length=5.0, overlap=0.5, class_label=None):
    """
    Liest PhyPhox-CSV-Daten ein und teilt sie in Segmente mit optionalem Überlappungsbereich
    
    Args:
        file_path: Pfad zur CSV-Datei
        segment_length: Länge jedes Segments in Sekunden
        overlap: Überlappungsanteil zwischen benachbarten Segmenten (0-1)
        class_label: Klassenlabel für diese Daten
        
    Returns:
        Liste von Segmenten und zugehörigen Labels
    """
    # CSV-Datei einlesen
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Fehler beim Einlesen von {file_path}: {e}")
        return [], []
    
    # Spaltennamen überprüfen
    time_col = "Time (s)"
    feature_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    # Sicherstellen, dass die erforderlichen Spalten vorhanden sind
    for col in [time_col] + feature_cols:
        if col not in data.columns:
            raise ValueError(f"Erforderliche Spalte '{col}' nicht in den Daten gefunden: {file_path}")
    
    # Daten nach Zeit sortieren (falls sie nicht bereits sortiert sind)
    data = data.sort_values(by=time_col)
    
    # Überprüfen der Abtastrate
    sampling_rate = 1 / np.median(np.diff(data[time_col]))
    print(f"Erkannte Abtastrate: {sampling_rate:.2f} Hz")
    
    # Berechnen der Schrittweite für überlappende Fenster
    stride = segment_length * (1 - overlap)
    
    # Gesamtzeit der Aufnahme bestimmen
    total_time = data[time_col].max() - data[time_col].min()
    start_time = data[time_col].min()
    
    # Berechnen der Anzahl der Segmente
    num_segments = max(1, int((total_time - segment_length) / stride) + 1)
    
    # Minimale Anzahl von Punkten pro Segment berechnen (basierend auf der Abtastrate)
    min_points = int(segment_length * sampling_rate * 0.8)  # Mindestens 80% der erwarteten Punkte
    
    print(f"Erstelle {num_segments} Segmente mit min. {min_points} Punkten pro Segment")
    
    segments = []
    labels = []
    
    for i in range(num_segments):
        segment_start = start_time + i * stride
        segment_end = segment_start + segment_length
        
        # Daten für dieses Segment extrahieren
        segment_data = data[(data[time_col] >= segment_start) & 
                           (data[time_col] < segment_end)]
        
        # Nur berücksichtigen, wenn genügend Datenpunkte vorhanden sind
        if len(segment_data) >= min_points:
            # Feature-Werte extrahieren
            segment_features = segment_data[feature_cols].values
            segments.append(segment_features)
            labels.append(class_label)
    
    print(f"Extrahierte {len(segments)} gültige Segmente aus {file_path}")
    return segments, labels

# Erweiterte Datenverarbeitungsfunktion mit Normalisierung
def process_data_files(class0_file, class1_file, segment_length=1.0, overlap=0.5):
    """
    Verarbeitet zwei Klassendateien und bereitet die Daten für das Training vor
    
    Args:
        class0_file: Pfad zur CSV-Datei für Klasse 0 (flacher Boden)
        class1_file: Pfad zur CSV-Datei für Klasse 1 (Treppe)
        segment_length: Länge jedes Segments in Sekunden
        overlap: Überlappungsanteil zwischen Segmenten
    
    Returns:
        Verarbeitete Daten X und Labels y
    """
    print(f"Verarbeite Klasse 0 (flacher Boden): {class0_file}")
    segments_class0, labels_class0 = read_phyphox_segments(
        class0_file, segment_length=segment_length, overlap=overlap, class_label=0
    )
    
    print(f"Verarbeite Klasse 1 (Treppe): {class1_file}")
    segments_class1, labels_class1 = read_phyphox_segments(
        class1_file, segment_length=segment_length, overlap=overlap, class_label=1
    )
    
    # Segmente und Labels kombinieren
    all_segments = segments_class0 + segments_class1
    all_labels = labels_class0 + labels_class1
    
    if not all_segments:
        raise ValueError("Keine gültigen Segmente gefunden. Überprüfen Sie die Daten und Parameter.")
    
    print(f"Gesamtzahl der Segmente: {len(all_segments)} (Klasse 0: {len(segments_class0)}, Klasse 1: {len(segments_class1)})")
    
    # Einheitliche Länge für alle Segmente finden
    # Anstatt auf die kleinste Länge zu reduzieren, können wir auch auf eine fixe Länge interpolieren
    # oder auf die häufigste Länge reduzieren
    
    # Option 1: Auf die mediane Länge zuschneiden/auffüllen (besser als Minimum)
    lengths = [segment.shape[0] for segment in all_segments]
    target_length = int(np.median(lengths))
    print(f"Standardisiere alle Segmente auf {target_length} Zeitpunkte")
    
    # Segmente standardisieren
    standardized_segments = []
    for segment in all_segments:
        if segment.shape[0] > target_length:
            # Zuschneiden
            standardized_segments.append(segment[:target_length])
        elif segment.shape[0] < target_length:
            # Auffüllen mit Nullen (oder bessere Strategie wie Interpolation verwenden)
            padded = np.zeros((target_length, segment.shape[1]))
            padded[:segment.shape[0]] = segment
            standardized_segments.append(padded)
        else:
            standardized_segments.append(segment)
    
    X = np.array(standardized_segments)
    y = np.array(all_labels)
    
    # Normalisierung der Daten (wichtig für Transformer)
    # Reshape für die Normalisierung
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    
    # Zurück zur ursprünglichen Form
    X = X_normalized.reshape(original_shape)
    
    print(f"Finale Datenform: {X.shape}, Labels: {y.shape}")
    return X, y

def create_model(input_shape, n_classes=2):
    """
    Erstellt ein optimiertes Transformer-Modell für die Bewegungsklassifikation
    
    Args:
        input_shape: Form der Eingabedaten
        n_classes: Anzahl der Klassen
    
    Returns:
        Kompiliertes Keras-Modell
    """
    # Hyperparameter für das Transformer-Modell
    head_size = 64  # Dimension der Attention-Heads
    num_heads = 4   # Anzahl der Attention-Heads
    ff_dim = 128    # Dimension des Feed-Forward-Netzwerks
    num_transformer_blocks = 3  # Anzahl der Transformer-Blöcke
    mlp_units = [64, 32]  # Einheiten im MLP-Teil
    dropout_rate = 0.2
    mlp_dropout = 0.3
    
    # Transformer-Encoder-Block-Definition
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    # Modell erstellen
    inputs = keras.Input(shape=input_shape)
    
    # Optional: Einbettungsschicht für bessere Repräsentation
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    
    # Transformer-Blöcke
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
    
    # Aggregation über die Zeitdimension
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    # MLP-Klassifikationskopf
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # Ausgabeschicht
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Kompilieren mit optimiertem Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["sparse_categorical_accuracy"]
    )
    
    return model

def train_and_evaluate(X, y, epochs=100, batch_size=32):
    """
    Trainiert und evaluiert das Modell mit Kreuzvalidierung
    
    Args:
        X: Feature-Daten
        y: Labels
        epochs: Maximale Anzahl der Trainingsepochen
        batch_size: Batch-Größe
        
    Returns:
        Trainiertes Modell und Trainingshistorie
    """
    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Trainings- und Validierungsdaten weiter aufteilen
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Trainingsset: {X_train.shape}, Validierungsset: {X_val.shape}, Testset: {X_test.shape}")
    
    # Modell erstellen
    model = create_model(input_shape=X_train.shape[1:])
    model.summary()


    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))

    
    # Callbacks für Training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy', 
            patience=20, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluierung
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Visualisierung der Trainingshistorie
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Speichere das trainierte Modell
    model.save("phyphox_activity_classifier.keras")
    
    return model, history

def main():
    # Konfiguration
    class0_file = "LinearAcceleration_flat.csv"  # flacher Boden
    class1_file = "LinearAcceleration.csv"       # Treppen
    
    # Optimale Parameter für die Segmentierung
    segment_length = 1.0  # 2 Sekunden pro Segment
    overlap = 0.5        # 50% Überlappung
    
    # Daten laden und vorverarbeiten
    X, y = process_data_files(
        class0_file, 
        class1_file, 
        segment_length=segment_length, 
        overlap=overlap
    )
    
    # Trainings- und Evaluierungsparameter
    epochs = 10000
    batch_size = 1
    
    # Modell trainieren und evaluieren
    model, history = train_and_evaluate(X, y, epochs=epochs, batch_size=batch_size)
    
    print("Training abgeschlossen. Das Modell wurde als 'phyphox_activity_classifier.keras' gespeichert.")
    
    return model, history

if __name__ == "__main__":
    main()
