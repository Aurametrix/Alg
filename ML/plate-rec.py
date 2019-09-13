# The dataset has to be in the FSNS dataset format. 
# For this, your test and train tfrecords along with the 
#charset labels text file are placed inside a folder named 
# 'fsns' inside the 'datasets' directory.

# you can change this to another folder and upload your
# tfrecord files and charset-labels.txt here. You'll
# have to change the path in multiple places accordingly. 
# I have used a directory called 'number_plates' inside 
# the datasets/data directory.
DATA_PATH = 'models/research/attention_ocr/python/datasets/data/number_plates'

## Now generate tf records by running the following script:

import os
import cv2
import random
import numpy as np 
import pandas as pd
import tensorflow as tf
from helpers import get_char_mapping
from tensorflow.python.platform import gfile

MAX_STR_LEN = 20

def read_image(img_path):
	return cv2.imread(img_path)

# Null ID depends on your charset label map. 
null = 43
def padding_char_ids(char_ids_unpadded, null_id = null, max_str_len=MAX_STR_LEN):
	return char_ids_unpadded + [null_id for x in range(max_str_len - len(char_ids_unpadded))]

def get_bytelist_feature(x):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=x))

def get_floatlist_feature(x):
	return tf.train.Feature(float_list = tf.train.FloatList(value=x))

def get_intlist_feature(x):
	return tf.train.Feature(int64_list = tf.train.Int64List(value=x))

def get_tf_example(img_file, annotation, num_of_views=1):

	img_array = read_image(img_file)
	img = gfile.FastGFile(img_file, 'rb').read()
	char_map, _ = get_char_mapping()

	text = annotation['text'].values[0]
	split_text = [x for x in text]
	char_ids_unpadded = [char_map[x] for x in split_text]
	char_ids_padded = padding_char_ids(char_ids_unpadded)
	char_ids_unpadded = [int(x) for x in char_ids_unpadded]
	char_ids_padded = [int(x) for x in char_ids_padded]

	features = tf.train.Features(feature = {
	'image/format': get_bytelist_feature([b'png']),
	'image/encoded': get_bytelist_feature([img]),
	'image/class': get_intlist_feature(char_ids_padded),
	'image/unpadded_class': get_intlist_feature(char_ids_unpadded),
	'image/width': get_intlist_feature([img_array.shape[1]]),
	'image/orig_width': get_intlist_feature([img_array.shape[1]/num_of_views]),
	'image/text': get_bytelist_feature([text])
		}
	)
	example = tf.train.Example(features=features)

	return example

def get_tf_records():

	train_file = DATA_PATH + '/' + 'train.tfrecord'
	test_file = DATA_PATH + '/' + 'test.tfrecord'
	if os.path.exists(train_file):
		os.remove(train_file)
	if os.path.exists(test_file):
		os.remove(test_file)
	train_writer = tf.io.TFRecordWriter(train_file)
	test_writer = tf.io.TFRecordWriter(test_file)
	annot = pd.read_csv(ANNOTATION_FILE) # define the annotation file explicitly
	annot['files'] = CROP_DIR + '/' + annot['files']
	files = list(annot['files'].values)
	random.shuffle(files)

	for i, file in enumerate(files):
		print('writing file:', file)
		annotation = annot.[annot['files'] == file]
		example = get_tf_example(file, annotation)
		if i < 251:
			train_writer.write(example.SerializeToString())
		else:
			test_writer.write(example.SerializeToString())

	train_writer.close()
	test_writer.close()

# Generate tfrecords!
if __name__ == '__main__':
    get_tf_records()

## Setting our Attention-OCR up
## Once we have our tfrecords and charset labels stored in the required directory, we need to write a dataset config script that will help us split our data into train and test for the attention OCR training script to process.

## Make a python file and name it 'number_plates.py' and place it inside the following directory:
### 'models/research/attention_ocr/python/datasets'
## The contents of the number-plates.py are as follows:

import datasets.fsns as fsns

DEFAULT_DATASET_DIR = 'models/research/attention_ocr/python/datasets/data/number_plates'

DEFAULT_CONFIG = {
    'name':
        'number_plates', # you can change the name if you want.
    'splits': {
        'train': {
            'size': 250, # change according to your own train-test split
            'pattern': 'train.tfrecord'
        },
        'test': {
            'size': 49, # change according to your own train-test split
            'pattern': 'test.tfrecord'
        }
    },
    'charset_filename':
        'charset-labels.txt',
    'image_shape': (200, 200, 3), # change this according to crop images size.
    'num_of_views':
        1,
    'max_sequence_length':
        MAX_STR_LEN, # TO BE CONFIGURED
    'null_code':
        43,
    'items_to_descriptions': {
        'image':
            'A (200X200) 3 channel color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config)

## Train the model
## Move into the following directory:
### models/research/attention_ocr

## Open the file named 'common_flags.py' and specify where you'd want to log your training.


# The train logs directory defaults to /tmp/attention_ocr/train. 
# You can change it to whatever you like.

LOGS_DIR = 'models/research/attention_ocr/number_plates_model_logs'
flags.DEFINE_string('train_log_dir', LOGS_DIR,
                      'Directory where to write event logs.')
and run the following command on your terminal:

# change this if you changed the dataset name in the 
# number_plates.py script or if you want to change the
# number of epochs

python train.py --dataset_name=number_plates --max_number_of_steps=3000
## Evaluate the model
## Run the following command from terminal.

python eval.py --dataset_name='number_plates'
## Get predictions
## Now from the same directory run the following command on your shell.

python demo_inference.py --dataset_name=number_plates --batch_size=8, \
--checkpoint='models/research/attention_ocr/number_plates_model_logs/model.ckpt-6000', \
--image_path_pattern=/home/anuj/crops/%d.png
## sweating off meme

## Using the GUI: https://app.nanonets.com/



## Alternatively - Using NanoNets API

## Step 1: Clone the Repo
## git clone https://github.com/NanoNets/nanonets-ocr-sample-python
## cd nanonets-ocr-sample-python
## sudo pip install requests
## sudo pip install tqdm
## Step 2: Get your free API Key
## Get your free API Key from https://app.nanonets.com/#/keys


## Step 3: Set the API key as an Environment Variable
export NANONETS_API_KEY=YOUR_API_KEY_GOES_HERE
## Step 4: Create a New Model
python ./code/create-model.py
## Note: This generates a MODEL_ID that you need for the next step

## Step 5: Add Model Id as Environment Variable
## export NANONETS_MODEL_ID=YOUR_MODEL_ID
## Step 6: Upload the Training Data
## Collect the images of object you want to detect. Once you have dataset ready in folder images (image files), start uploading the dataset.

python ./code/upload-training.py
## Step 7: Train Model
## Once the Images have been uploaded, begin training the Model

python ./code/train-model.py
## Step 8: Get Model State
## The model takes ~30 minutes to train. You will get an email once the model is trained. In the meanwhile you check the state of the model

watch -n 100 python ./code/model-state.py
## Step 9: Make Prediction
## Once the model is trained. You can make predictions using the model

python ./code/prediction.py PATH_TO_YOUR_IMAGE.jpg

