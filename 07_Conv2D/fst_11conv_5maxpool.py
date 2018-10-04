import threading
import time
import sys
import os
import logging
import tensorflow as tf

from keras.callbacks import Callback
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Add, Merge, Flatten, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime, timedelta


ds_width = 128
ds_height = 128
output_dim = 7

_SAVE_MODEL_DIR_ = 'graduated'
_LOAD_MODEL_DIR_ = 'model'

if not os.path.exists(_SAVE_MODEL_DIR_): os.makedirs(_SAVE_MODEL_DIR_)
_LOG_DIR_ = 'logs'
if not os.path.exists(_LOG_DIR_): os.makedirs(_LOG_DIR_)

logging.basicConfig(filename='%s/%s.log' % (_LOG_DIR_, datetime.today().strftime('%Y%m%d')), level=logging.INFO)

_PROGRESS_MSG_ = ''
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
	global _PROGRESS_MSG_

	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)

	_PROGRESS_MSG_ = '%s %s%s %s' % (prefix, percent, '%', suffix)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

def seperator():
    print('-' * 120)

class IDGAgent:
	def __init__(self):
		self.model = self.build_model()
		# self.model = self.load_model()
		self.model.load_weights(self.load_weights())
		# self.save_model() - Model saving on training begin.

		# print(self.model.summary())
		# exit()

	def load_model(self):
		# load json and create model
		json_file = open('%s/_model_.json' % _LOAD_MODEL_DIR_, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		return model

	def load_weights(self):
		return '%s/_model_weight_.h5' % _LOAD_MODEL_DIR_

	def build_model(self):
		# from tensorflow.python.client import device_lib
		# print(device_lib.list_local_devices())
		# exit()

		img_input = Input(shape=(ds_height, ds_width, 1))

		# ===========ENTRY FLOW==============
		with tf.device('/gpu:0'):
			# Block 01 ---
			x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(img_input)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = Conv2D(64, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)

		with tf.device('/gpu:3'):
			y = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
			y = BatchNormalization()(y)

		with tf.device('/gpu:1'):
			# Block 02 ---
			x = SeparableConv2D(128, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = SeparableConv2D(128, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
			x = Add()([x, y])

		with tf.device('/gpu:3'):
			y = Conv2D(256, (1, 1), strides=(2, 2), padding='same')(x)
			y = BatchNormalization()(y)

		with tf.device('/gpu:2'):
			x = Activation('relu')(x)

			# Block 03 ---
			x = SeparableConv2D(256, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = SeparableConv2D(256, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
			x = Add()([x, y])

		with tf.device('/gpu:3'):
			y = Conv2D(512, (1, 1), strides=(2, 2), padding='same')(x)
			y = BatchNormalization()(y)

		with tf.device('/gpu:0'):
			x = Activation('relu')(x)

			# Block 04 ---
			x = SeparableConv2D(512, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = SeparableConv2D(512, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
			x = Add()([x, y])

		with tf.device('/gpu:3'):
			y = Conv2D(1024, (1, 1), strides=(2, 2), padding='same')(x)
			y = BatchNormalization()(y)

		with tf.device('/gpu:2'):
			x = Activation('relu')(x)

			# Block 05 ---
			x = SeparableConv2D(1024, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = SeparableConv2D(1024, (3, 3), padding='same')(x)
			x = BatchNormalization()(x)
			x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
			x = Add()([x, y])

		with tf.device('/gpu:3'):
			x = Activation('relu')(x)

			x = GlobalAveragePooling2D(name='avg_pool')(x)
			x = Dense(output_dim, activation='softmax')(x)

		# Create model.
		model = Model(img_input, x)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		return model


	def save_model(self, model_located):
		# serialize model to JSON
		json_model = self.model.to_json()

		with open(model_located, "w") as json_file:
			json_file.write(json_model)

	def save_weight(self, file_name):
		self.model.save_weights(file_name)

	def train(self):
		epochs = 100
		batch_size = 512

		with tf.device('/cpu:0'):
			train_datagen = ImageDataGenerator(rescale=1./255)
			train_generator = train_datagen.flow_from_directory(
				'plots',
				target_size=(ds_height, ds_width),
				batch_size=batch_size,
				color_mode='grayscale',
				class_mode='categorical'
			)
			"""
			train_generator = train_datagen.flow_from_directory(
				'plots',
				target_size=(ds_height, ds_width),
				batch_size=1,
				color_mode='grayscale',
				class_mode='categorical'
			)
			image, label = next(train_generator)
			print(image.shape, label)
			exit()
			"""

			# monitor = Monitor(self)
			# monitor.start()

			steps_per_epoch = int(train_generator.n / batch_size) + 1
			verbose = 0

			learing_process = LearningProcess(self, train_generator, epochs, steps_per_epoch)
			self.model.fit_generator(
				train_generator,
				steps_per_epoch=steps_per_epoch,
				epochs=epochs,
				verbose=verbose,
				max_queue_size=batch_size,
				workers=4,
				use_multiprocessing=True,
				callbacks = [learing_process]
			)
			"""
			self.model.fit_generator(
				train_generator,
				steps_per_epoch=192,
				epochs=50
			)
			"""
			"""
			monitor.stopped = True
			while True:
				time.sleep(5)
				if monitor.terminated:
					monitor.stop()
					break
			"""


class LearningProcess(Callback):
	def __init__(self, creator, gen, epoch, steps_per_epoch):
		super(Callback, self).__init__()
		self.creator = creator
		self.gen = gen
		self.epoch_total = epoch
		self.steps_per_epoch = steps_per_epoch
		self.batch_current = 0
		self.batch_total = steps_per_epoch * epoch
		self.aucs = []
		self.losses = []


	def save_model(self):
		model_located = '%s/_model_%s.json' % (_SAVE_MODEL_DIR_, datetime.today().strftime('%Y%m%d'))
		self.creator.save_model(model_located)

	def save_weight(self, event_name, acc):
		weight_located = '%s/_model_weight_%s_%s_%s.h5' % (_SAVE_MODEL_DIR_, event_name, acc, datetime.today().strftime('%Y%m%d'))
		self.creator.save_weight(weight_located)

	def on_train_begin(self, logs={}):
		self.time_train_begin = datetime.today()
		self.batch_last_loged = datetime.today()
		self.batch_last_saved = datetime.today()

		seperator()
		printstr = '%s%s%s%s' % (
			'##### TRAINING INFORMATION SUMMARY. ',
			'\n    - TRAINING SIZE: %s to be %s epoches by every %s per batch of %s batches.' % (
				'{0:,d}'.format(self.gen.n),
				self.epoch_total,
				self.gen.batch_size,
				self.steps_per_epoch
			),
			'\n    - START TIME   : ',
			self.time_train_begin.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
		)

		print(printstr)
		seperator()

		self.save_model()
		logging.info(printstr)

	def on_train_end(self, logs={}):
		end_time = datetime.today()

		seperator()
		printstr = '%s%s%s%s%s%s' % (
			'##### TRAINING INFORMATION SUMMARY. ',
			'\n    - TRAINING SIZE    : %s to be %s epoches by every %s per batch of %s batches.' % (
				'{0:,d}'.format(self.gen.n),
				self.epoch_total,
				self.gen.batch_size,
				self.steps_per_epoch
			),
			'\n    - END TIME         :', end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
			'\n    - TOTAL SPENT TIME :', str(end_time - self.time_train_begin)[:-3]
		)
		print(printstr)
		seperator()

		logging.info(printstr)

	def on_epoch_begin(self, epoch, logs={}):
		self.time_epoch_being = datetime.today()
		self.epoch = epoch + 1

	def on_epoch_end(self, epoch, logs={}):
		end_time = datetime.today()

		seperator()
		printstr = '%s%s' % (
			'##### [%s] PROGESS:' % end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
			'(EPOCH: %s / %s) COMPLETED. ### LOSS&ACCURACY: [%s, %s] ### %s / %s / %s' % (
				self.epoch,
				self.epoch_total,
				'{0:.8f}'.format(logs.get('loss')),
				'{0:.2f}'.format(logs.get('acc')),
				self.time_epoch_being.strftime('%Y-%m-%d %H:%M:%S'),
				end_time.strftime('%Y-%m-%d %H:%M:%S'),
				str(end_time - self.time_epoch_being)[:-3]
			)
		)
		print(printstr)
		seperator()

		logging.info(printstr)

		self.save_weight('epoch', int(logs.get('acc')*100))


	def on_batch_begin(self, batch, logs={}):
		self.time_batch_begin = datetime.today()

	def on_batch_end(self, batch, logs={}):
		end_time = datetime.today()

		current_proc = batch * self.gen.batch_size
		current_proc = self.gen.n if self.gen.n < current_proc else current_proc

		printProgress(
			self.batch_current * self.gen.batch_size,
			self.gen.n * self.epoch_total,
			'##### [%s] PROGESS:' % end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
			'(CURRENT: %s, BATCH: %s / %s, EPOCH: %s / %s) COMPLETED. ### LOSS&ACCURACY: [%s, %s] ### TIME SPENT: %s' % (
				'{0:3.4f}'.format(current_proc / self.gen.n * 100) + '%',
				self.batch_current + 1,
				self.batch_total,
				self.epoch,
				self.epoch_total,
				'{0:2.8f}'.format(logs.get('loss')),
				'{0:.2f}'.format(logs.get('acc')),
				str(end_time - self.time_batch_begin)[:-3] + ('\n' if (batch + 1) % self.steps_per_epoch == 0 else '')
			), 4, 30
		)

		if end_time - timedelta(minutes=10) > self.batch_last_loged:
			self.batch_last_loged = end_time
			logging.info(_PROGRESS_MSG_)

		if end_time - timedelta(hours=1) > self.batch_last_saved:
			self.batch_last_saved = end_time
			self.save_weight('batch', int(logs.get('acc') * 100))

		self.batch_current += 1


class Monitor(threading.Thread):
	def __init__(self, creator):
		threading.Thread.__init__(self)

		self.CREATOR = creator

		self.terminated = False
		self.stopped = False
		self._stop_event = threading.Event()

	def run(self):
		# time.sleep(60 * 60 * 24)

		while not self.stopped:
			self.CREATOR.save_weight()
			time.sleep(60)

		self.terminated = True

	def stop(self):
		self._stop_event.set()
		print('##### TERMINATING : %s' % self)

		try:
			self._stop()
		except:
			pass

	def is_stopped(self):
		return self._stop_event.is_set()


if __name__ == "__main__":
	agent = IDGAgent()
	agent.train()
