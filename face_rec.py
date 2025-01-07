import cv2
import os
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
user_id = input("Enter User ID (e.g., 1 for Chandu, 2 for Cherry): ") #create your desired face data collection
user_name = input("Enter your name: ")

user_dir = os.path.join('dataset', user_name)
os.makedirs(user_dir, exist_ok=True)

print("\nLook at the camera and wait for image collection...")

count = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1


        cv2.imwrite(f"{user_dir}/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w]) #saving the image

        cv2.imshow('image', img)

    
    if cv2.waitKey(100) & 0xFF == ord('q') or count >= 30:
        break

print("\nImage collection complete!")
cam.release()
cv2.destroyAllWindows()



#data preprocessing
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical  
from sklearn.model_selection import train_test_split


dataset_path = 'dataset'
image_size = (100, 100)

def load_data(dataset_path):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for label_name in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label_name)
        if os.path.isdir(label_folder):
            label_map[label_id] = label_name
            for img_file in os.listdir(label_folder):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(label_folder, img_file)
                    img = load_img(img_path, target_size=image_size, color_mode='grayscale')
                    img = img_to_array(img) / 255.0  #image normalize
                    images.append(img)
                    labels.append(label_id)
            label_id += 1

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_map


X, y, label_map = load_data(dataset_path) #loading data


print(f"Loaded {len(X)} images with {len(label_map)} unique labels.") #checking the data 

# Converting labels to one-hot encoding
if len(y) > 0:  # Ensure y is not empty
    y = to_categorical(y)
else:
    raise ValueError("No data found. Please ensure your dataset directory is not empty and correctly structured.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





#model training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (100, 100, 1)  # Image shape
num_classes = len(label_map)  # Number of unique labels

# Building and compile the model
model = build_cnn_model(input_shape, num_classes)
model.summary()


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save('face_recognition_model.h5')
#recognising
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Loading the trained model
model = load_model('face_recognition_model.h5')

# Loading the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = np.expand_dims(face, axis=-1)  # Adding channel dimension
        face = np.expand_dims(face, axis=0)   # Adding batch dimension

        # Predict the label
        predictions = model.predict(face)
        label = np.argmax(predictions)
        confidence = np.max(predictions)

        # Get the name from the label map
        name = label_map[label]
        confidence_text = f"{round(confidence * 100, 2)}%"

        # Drawingg the rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence_text})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame 
    cv2.imshow('Face Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

