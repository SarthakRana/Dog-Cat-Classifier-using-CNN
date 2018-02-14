# Convolutional Neural Network

# Part 1 - Building the CNN.
# Importing keras libraries and packages
from keras.models import Sequential # Sequential is used to initialize our NN (sequence of layers)
from keras.layers import Dense   # Dense is used to add a new layer in NN
from keras.layers import Convolution2D   # Convolution2D since images are in 2D and not 3D
from keras.layers import MaxPooling2D   #Pooling the feature maps
from keras.layers import Flatten   #Flatten out the pooled feature maps to input vector.

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution layer
# 32 3 3 means we are gonna create 32 feature detectors of 3x3 size each
# input_shape(shape of ur input image on which we apply feature detectors) = 3 - here 3 is the
# format(number of channels(RBG)/1 if b/w image) and each channel has 64x64 2D array from image. 
classifier.add(Convolution2D(32, (3, 3), input_shape = (32, 32, 3), activation = "relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer (for better accuracy in test set)
classifier.add(Convolution2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 -Flattening
classifier.add(Flatten())

# Step 4 -Full Connection
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
"""Image augmentation which consists of preprocessing our images to prevent overfitting. 
Overfitting occurs when we have few data to train(8000 in this case which is very less) coz 
our model finds some correlation in the few obsv. of the training set but fails to generalize 
this correlation on some new obsv. Image augmentation will create many batches of images and 
in each batch apply some random transformations like rotating, shifting, rotating and thus we 
get many more images to train. SUMMARY - Img. augmentation is a technique which allows us to 
enrich our dataset without adding images and thus gives us good performance results with no 
overfitting even with a small amt. of images."""

"""ImageDataGenerator will perform the img. augmentation. Next two functions will create the 
train set and the test set respectively. The fit generator method will fit the CNN with our 
trainig set and at the same time will test the performance on the test set"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,         # Feature scaling(pixel values = (0-255) -> (0-1))
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',           # directory
        target_size=(32, 32),           # 64 is the size of 2D array of image
        batch_size=32,         #No of imgs CNN will go thru after which the weights will be updated.
        class_mode='binary')     # Two categories only - cat n dog.

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
            training_set,            
            steps_per_epoch=8000,       #No. of imgs we have in our training set
            epochs=25,
            validation_data=test_set,
            validation_steps=2000)       #No. of imgs in our test set