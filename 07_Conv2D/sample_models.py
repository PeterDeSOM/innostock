import threading
import time
import sys
import os
import logging

from keras.callbacks import Callback
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Add, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime, timedelta

# 01. Convolutional 2D Xcenption ######################################################################################
ds_width = 256
ds_height = 256
output_dim = 7

_MODEL_DIR_ = 'graduated'
if not os.path.exists(_MODEL_DIR_): os.makedirs(_MODEL_DIR_)
_LOG_DIR_ = 'logs'
if not os.path.exists(_LOG_DIR_): os.makedirs(_LOG_DIR_)

logging.basicConfig(filename='%s/%s.log' % (_LOG_DIR_, datetime.today().strftime('%Y%m%d')), level=logging.INFO)

_PROGRESS_MSG_ = ''

def build_model_xception(self):
	img_input = Input(shape=(ds_height, ds_width, 1))

	# ===========ENTRY FLOW==============
	# Block 01 ---
	x = Conv2D(32, (3, 3), strides=(2, 2))(img_input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(64, (3, 3))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	y = Conv2D(128, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	# Block 02 ---
	x = SeparableConv2D(128, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(128, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = Add()([x, y])

	y = Conv2D(256, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	# Block 03 ---
	x = SeparableConv2D(256, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = Add()([x, y])

	y = Conv2D(512, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	# Block 04 ---
	x = SeparableConv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = Add()([x, y])

	y = Conv2D(1024, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	# Block 05 ---
	x = SeparableConv2D(1024, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(1024, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = Add()([x, y])

	# ===========MIDDLE FLOW===============
	for i in range(9):
		y = x
		x = Activation('relu')(x)
		x = SeparableConv2D(1024, (3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = SeparableConv2D(1024, (3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = SeparableConv2D(1024, (3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = Add()([x, y])

	# ========EXIT FLOW============
	y = Conv2D(2048, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	x = SeparableConv2D(1024, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(2048, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = Add()([x, y])

	x = SeparableConv2D(2048, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(4096, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = GlobalAveragePooling2D(name='avg_pool')(x)
	x = Dense(output_dim, activation='softmax')(x)

	# Create model.
	model = Model(img_input, x)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

# 01. Convolutional 2D Xcenption ######################################################################################
def build_model_stockchart(self):
	img_input = Input(shape=(ds_height, ds_width, 1))

	# ===========ENTRY FLOW==============
	# Block 01 ---
	x = Conv2D(32, (3, 3), strides=(2, 2))(img_input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(64, (3, 3))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	y = Conv2D(128, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	# Block 02 ---
	x = SeparableConv2D(128, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(128, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Add()([x, y])

	y = Conv2D(256, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	# Block 03 ---
	x = SeparableConv2D(256, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Add()([x, y])

	y = Conv2D(512, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	# Block 04 ---
	x = SeparableConv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(512, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Add()([x, y])

	y = Conv2D(1024, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	# Block 05 ---
	x = SeparableConv2D(1024, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(1024, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Add()([x, y])

	# ===========MIDDLE FLOW===============
	for i in range(9):
		y = x
		x = Activation('relu')(x)
		x = SeparableConv2D(1024, (3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = SeparableConv2D(1024, (3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = SeparableConv2D(1024, (3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = Add()([x, y])

	# ========EXIT FLOW============
	y = Conv2D(2048, (1, 1), strides=(2, 2))(x)
	y = BatchNormalization()(y)

	x = Activation('relu')(x)

	x = SeparableConv2D(1024, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(2048, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Add()([x, y])

	x = SeparableConv2D(2048, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(4096, (3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Flatten(x)
	x = Dense(output_dim, activation='softmax')(x)

	# Create model.
	model = Model(img_input, x)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
