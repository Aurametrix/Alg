### https://github.com/matterport/Mask_RCNN/blob/master/coco.py

# Train a new model starting from pre-trained COCO weights
python3 coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 coco.py train --dataset=/path/to/coco/ --model=last
