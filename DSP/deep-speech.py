# Project DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques, 
# based on Baidu's Deep Speech research paper: https://arxiv.org/abs/1412.5567
# Project DeepSpeech uses Google's TensorFlow project to make the implementation easier.

# if using pre-trained model:
wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.1.1/deepspeech-0.1.1-models.tar.gz | tar xvfz -

deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my_audio_file.wav
