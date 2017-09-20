
input_shape = (160,320,3)
#input_shape = (80, 320, 3)  # Trimmed image format
XXX = 3

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.image as mpimg




def model0():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape, output_shape=input_shape))
	model.add(Flatten())
	model.add(Dense(1))
	return model


def modelLeNet():
	model = Sequential()

	model.add( Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape, output_shape=input_shape))
	model.add( Cropping2D(cropping=((70,25),(0,0))) )

	model.add( Convolution2D(6,5,5,activation='relu') )
	model.add( MaxPooling2D() )
	model.add( Convolution2D(6,5,5,activation='relu') )
	model.add( MaxPooling2D() )
	model.add( Convolution2D(6,5,5,activation='relu') )
	model.add( MaxPooling2D() )
	
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

def modelNvidia():
	model = Sequential()

	model.add( Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape, output_shape=input_shape) )
	model.add( Cropping2D(cropping=((70,25),(0,0)) ))

	model.add( Convolution2D(24,5,5,subsample=(2,2),activation='relu') )
	model.add( Convolution2D(36,5,5,subsample=(2,2),activation='relu') )
	model.add( Convolution2D(48,5,5,subsample=(2,2),activation='relu') )
	model.add( Convolution2D(64,3,3,activation='relu') )
	model.add( Convolution2D(64,3,3,activation='relu') )

	model.add( MaxPooling2D() )
	model.add( Convolution2D(6,5,5,activation='relu') )
	model.add( MaxPooling2D() )
	model.add( Convolution2D(6,5,5,activation='relu') )
	model.add( MaxPooling2D() )
	
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model


def modelCommaAI(time_len=1):
	ch, row, col = 3, 160, 320  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=input_shape,
		output_shape=input_shape))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model



def read_csv_lines(path):
	lines = []
	with open(path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	return lines






# data
images = []
measurements = []

samples = read_csv_lines('./data/driving_log.csv')
del samples[0]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		np.random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for j in range(XXX):
					delta = j*0.2 if j<2 else -0.2

					name = './data/IMG/'+batch_sample[j].split('/')[-1]
					
					center_image = mpimg.imread(name)
					#center_image = cv2.imread(name)
					center_angle = float(batch_sample[3])

					center_angle = center_angle * delta

					images.append(center_image)
					angles.append(center_angle)

					images.append(cv2.flip(center_image,1))
					angles.append(-center_angle)


			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)






model = model0()
#model = modelLeNet()
#model = modelNvidia()
model.compile(loss='mse', optimizer='adam')


print('Training...')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epochs=2, verbose=1)
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 2 * XXX, validation_data = validation_generator, nb_val_samples = len(validation_samples) * 2 * XXX, nb_epoch=5, verbose=1)
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.hf5')



import matplotlib.pyplot as plt


### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



exit()