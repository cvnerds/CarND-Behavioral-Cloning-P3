
import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU

import matplotlib.image as mpimg


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('in', './data/driving_log.csv', "CSV logfile of driving data")
flags.DEFINE_string('out', 'model.h5', 'where to save the trained model')
flags.DEFINE_string('model', 'nvidia', "Choose the model: [simple, lenet, nvidia, commai]")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

# image dimensions
input_shape = (160,320,3)


# reads a CSV line-by-line and returns a list of lines
def read_csv_lines(path):
	lines = []
	with open(path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	return lines

# function to load the data from path
def load_data(path, test_split=0.2):
	samples = read_csv_lines(path)
	del samples[0]
	return train_test_split(samples, test_size=test_split)


# generator that yields training batches with augmentations
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				# use all three cameras
				for j in range(3):
					# offset 0/0.2/-0.2
					delta = j*0.2 if j<2 else -0.2

					name = './data/IMG/'+batch_sample[j].split('/')[-1]
					image = mpimg.imread(name)
					
					angle = float(batch_sample[3]) + delta

					# add data sample
					images.append(image)
					angles.append(angle)

					# flip sample to double and balance data
					images.append(np.fliplr(image))
					angles.append(-angle)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)




# def modelSimple():
# 	model = Sequential()
# 	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
# 	model.add(Flatten())
# 	model.add(Dense(1))
# 	return model


def modelLeNet():
	model = Sequential()
	model.add( Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
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
	model.add( Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape) )
	model.add( Cropping2D(cropping=((70,25),(0,0)) ))
	model.add( Convolution2D(24,5,5,subsample=(2,2),activation='relu') )
	model.add( Convolution2D(36,5,5,subsample=(2,2),activation='relu') )
	model.add( Convolution2D(48,5,5,subsample=(2,2),activation='relu') )
	model.add( Convolution2D(64,3,3,activation='relu') )
	model.add( Convolution2D(64,3,3,activation='relu') )
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model


def modelCommaAI(time_len=1):
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape))
	#model.add( Cropping2D(cropping=((70,25),(0,0)) ))
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



def main(_):
	train_samples, validation_samples = load_data('./data/driving_log.csv')

	# compile and train the model using the generator function
	train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
	validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

	models = {
		# 'simple': modelSimple,
		'lenet':  modelLeNet,
		'nvidia': modelNvidia,
		'commai': modelCommaAI
	}

	model = models[FLAGS.model]()
	model.compile(loss='mse', optimizer='adam')

	print('Training...')
	history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 2 * 3, validation_data = validation_generator, nb_val_samples = len(validation_samples) * 2 * 3, nb_epoch=FLAGS.epochs, verbose=1)
	model.save(FLAGS.output)


	print(history_object.history.keys())
	print('Loss')
	print(history_object.history['loss'])
	print('Validation Loss')
	print(history_object.history['val_loss'])

	# no visualization on AWS

	# # import matplotlib.pyplot as plt

	# ### print the keys contained in the history object
	# print(history_object.history.keys())
	# ### plot the training and validation loss for each epoch
	# plt.plot(history_object.history['loss'])
	# plt.plot(history_object.history['val_loss'])
	# plt.title('model mean squared error loss')
	# plt.ylabel('mean squared error loss')
	# plt.xlabel('epoch')
	# plt.legend(['training set', 'validation set'], loc='upper right')
	# plt.save()
	# #plt.show()




# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

