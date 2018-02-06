# Load the images and preprocess them for inception-resnet
images = []
all_filenames = listdir('images/')
all_filenames.sort()
for filename in all_filenames:
    images.append(img_to_array(load_img('images/'+filename, target_size=(299, 299))))
images = np.array(images, dtype=float)
images = preprocess_input(images)
# Run the images through inception-resnet and extract the features without the classification layer
IR2 = InceptionResNetV2(weights='imagenet', include_top=False)
features = IR2.predict(images)
# We will cap each input sequence to 100 tokens
max_caption_len = 100
# Initialize the function that will create our vocabulary 
tokenizer = Tokenizer(filters='', split=" ", lower=False)
# Read a document and return a string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
# Load all the HTML files
X = []
all_filenames = listdir('html/')
all_filenames.sort()
for filename in all_filenames:
    X.append(load_doc('html/'+filename))
# Create the vocabulary from the html files
tokenizer.fit_on_texts(X)
# Add +1 to leave space for empty words
vocab_size = len(tokenizer.word_index) + 1
# Translate each word in text file to the matching vocabulary index
sequences = tokenizer.texts_to_sequences(X)
# The longest HTML file
max_length = max(len(s) for s in sequences)
# Intialize our final input to the model
X, y, image_data = list(), list(), list()
for img_no, seq in enumerate(sequences):
    for i in range(1, len(seq)):
        # Add the entire sequence to the input and only keep the next word for the output
        in_seq, out_seq = seq[:i], seq[i]
        # If the sentence is shorter than max_length, fill it up with empty words
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # Map the output to one-hot encoding
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # Add and image corresponding to the HTML file
        image_data.append(features[img_no])
        # Cut the input sentence to 100 tokens, and add it to the input data
        X.append(in_seq[-100:])
        y.append(out_seq)
X, y, image_data = np.array(X), np.array(y), np.array(image_data)
# Create the encoder
image_features = Input(shape=(8, 8, 1536,))
image_flat = Flatten()(image_features)
image_flat = Dense(128, activation='relu')(image_flat)
ir2_out = RepeatVector(max_caption_len)(image_flat)
language_input = Input(shape=(max_caption_len,))
language_model = Embedding(vocab_size, 200, input_length=max_caption_len)(language_input)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = LSTM(256, return_sequences=True)(language_model)
language_model = TimeDistributed(Dense(128, activation='relu'))(language_model)
# Create the decoder
decoder = concatenate([ir2_out, language_model])
decoder = LSTM(512, return_sequences=False)(decoder)
decoder_output = Dense(vocab_size, activation='softmax')(decoder)
# Compile the model
model = Model(inputs=[image_features, language_input], outputs=decoder_output)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# Train the neural network
model.fit([image_data, X], y, batch_size=64, shuffle=False, epochs=2)
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'START'
    # iterate over the whole length of the sequence
    for i in range(900):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0][-100:]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # Print the prediction
        print(' ' + word, end='')
        # stop if we predict the end of the sequence
        if word == 'END':
            break
    return
# Load and image, preprocess it for IR2, extract features and generate the HTML
test_image = img_to_array(load_img('images/87.jpg', target_size=(299, 299)))
test_image = np.array(test_image, dtype=float)
test_image = preprocess_input(test_image)
test_features = IR2.predict(np.array([test_image]))
generate_desc(model, tokenizer, np.array(test_features), 100)
