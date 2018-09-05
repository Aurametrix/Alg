import snowy

source = snowy.open('poodle.png')
source = snowy.resize(source, height=200)
blurry = snowy.blur(source, radius=4.0)
snowy.save(snowy.hstack([source, blurry]), 'diptych.png')

# This snippet does a resize, then a blur, then horizontally concatenates the two images

parrot = snowy.load('parrot.png')
height, width = parrot.shape[:2]
nearest = snowy.resize(parrot, width * 6, filter=snowy.NEAREST) 
mitchell = snowy.resize(parrot, width * 6)
snowy.show(snowy.hstack([nearest, mitchell]))

#  This snippet first magnifies an image using a nearest-neighbor filter, then using the default Mitchell filter.

parrot = snowy.load('parrot.png')
height, width = parrot.shape[:2]
nearest = snowy.resize(parrot, width * 6, filter=snowy.NEAREST) 
mitchell = snowy.resize(parrot, width * 6)
snowy.show(snowy.hstack([nearest, mitchell]))
