import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class FaceRecognitionSystem:
    def __init__(self, input_shape=(224, 224, 3), n_classes=10):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.models = self.build_models()
        self.ensemble = self.build_ensemble()
        self.data_generator = self.build_data_generator()

    def build_models(self):
        models = {
            'resnet50': self.build_transfer_model(ResNet50, 'resnet50'),
            'siamese': self.build_siamese_model()
        }
        return models

    def build_transfer_model(self, base_model_class, name):
        base_model = base_model_class(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output, name=name)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_siamese_model(self):
        base_network = self.get_base_network()
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([processed_a, processed_b])
        output = Dense(1, activation='sigmoid')(distance)
        model = Model(inputs=[input_a, input_b], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def get_base_network(self):
        input = Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (7, 7), activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        return Model(inputs=input, outputs=x)

    def build_ensemble(self):
        estimators = [
            ('resnet50', self.models['resnet50']),
            ('svm', SVC(kernel='rbf', probability=True))
        ]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        return ensemble

    def build_data_generator(self):
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

    def train(self, X, y, epochs=20, batch_size=32):
        # Convert labels to categorical
        num_classes = len(np.unique(y))
        y_categorical = to_categorical(y, num_classes=num_classes)

        # Ensure the model's output layer matches the number of classes
        if num_classes != self.n_classes:
            print(f"Adjusting model for {num_classes} classes")
            self.n_classes = num_classes
            self.models = self.build_models()  # Rebuild models with correct number of classes

        X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

        # Train individual models
        for name, model in self.models.items():
            if name != 'siamese':
                print(f"Training {name} model...")
                model.fit(self.data_generator.flow(X_train, y_train, batch_size=batch_size),
                          validation_data=(X_val, y_val),
                          epochs=epochs)

        # Train Siamese model
        print("Training Siamese model...")
        self.train_siamese(X_train, np.argmax(y_train, axis=1), epochs, batch_size)

        # Train SVM
        print("Training SVM...")
        resnet_features = self.models['resnet50'].predict(X_train)
        self.ensemble.estimators_[1][1].fit(resnet_features, np.argmax(y_train, axis=1))

        # Train ensemble
        print("Training ensemble...")
        ensemble_features = np.hstack([model.predict(X_train) for name, model in self.models.items() if name != 'siamese'])
        self.ensemble.fit(ensemble_features, np.argmax(y_train, axis=1))

    def train_siamese(self, X, y, epochs, batch_size):
        def generate_pairs(X, y):
            pairs = []
            labels = []
            n_classes = len(np.unique(y))
            for i in range(len(X)):
                positive_idx = np.random.choice(np.where(y == y[i])[0])
                negative_idx = np.random.choice(np.where(y != y[i])[0])
                pairs += [[X[i], X[positive_idx]], [X[i], X[negative_idx]]]
                labels += [1, 0]
            return np.array(pairs), np.array(labels)

        pairs, pair_labels = generate_pairs(X, y)
        self.models['siamese'].fit([pairs[:, 0], pairs[:, 1]], pair_labels, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        ensemble_features = np.hstack([model.predict(X) for name, model in self.models.items() if name != 'siamese'])
        return self.ensemble.predict(ensemble_features)

    def recognize_face(self, face):
        face = cv2.resize(face, self.input_shape[:2])
        face = np.expand_dims(face, axis=0)
        prediction = self.predict(face)
        return np.argmax(prediction), np.max(prediction)

# Usage example
if __name__ == "__main__":
    # Load your dataset here
    X = np.random.rand(1000, 224, 224, 3)  # Replace with your actual image data
    y = np.random.randint(0, 10, 1000)  # Replace with your actual labels

    face_recognition_system = FaceRecognitionSystem(n_classes=len(np.unique(y)))
    face_recognition_system.train(X, y)

    # Test the system
    test_face = np.random.rand(224, 224, 3)  # Replace with an actual test image
    predicted_class, confidence = face_recognition_system.recognize_face(test_face)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
