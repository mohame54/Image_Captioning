# Image Captioning

**This project is about building an image captioning model using the Framework that we all love `TensorFlow` .**

**I assume you have some basic knowledge of tensorflow if you don't  I advise you to visit [TensorFlow](https://www.tensorflow.org/overview) .**

# Contents:
1. ***Objective***
2. ***Concepts***
3. ***Implementation***

# Objective:

**To build this kind of model that can generate a descriptive text about an image that we provide** 

*for the sake of keeping things easy to understand I will use a Transformer architecture that was mentioned  in [Attention is all you need](https://arxiv.org/abs/1706.03762).*
<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="drawing" style="width:400px"/>

*it was difficult at first to understand this masterpiece but later after I saw the implementation of the attention mechanism in recurrent neural nets, I was convinced that the Transformer was the right architecture for building models used in natural language processing .*

*You can find the source code in [here](https://keras.io/examples/vision/image_captioning/) .*

*The model learns well how and where to look in a specific image, using the attention mechanism .*

*As  the Transformer generate word by word moving its attention from portion to another in the image .*

# Concepts:
* **image captioning duh .**
* **Transfer learning:** A technique used to benefit from models that were trained on problems and use the knowledge that these models learned to apply it to another but related problems for instance: ( knowledge gained from model used to recognise cars could also be used to recognise trucks) .  
* **Cnn model:** A model used to extract  useful features from image using (Transfer learning) :wink: . 
* **Encoder, Decoder:** Main parts of Transformer model that encode the  learned features then decode  them back to words .
* **Image caption module:** A Tensorflow Module that can take a path of the image then outputs the caption . 
* **Tokenizer:** An object  used to transform plain text into sequence of integers that every sequence presents a text .

# Implementation:
**Dataset:** I used `MSCOCO`  as my training dataset with size of 13GB for training and 6GB for validation, but I used a small portion of it .

**The inputs:** The inputs are images and sequences, images converted into `float32 dtype` instead of `uint8 dtype` then its resized into (299,299,3) which is suitable for `EffcientNetB0` (the model used to extract the features)
and sequences which are transformed form of the captions cuz the computer doesn't understand text only numbers .

**Data pipeline:** The images captions are stored in a json file after extracting the captions and mapping each caption to its corresponding image then passing
the mapping dict to `preprocess_inputs.py` which contains some utility functions to make the dataset (instance of tensorflow.data.Datasets),
prepare the tokenizer object for processing text and getting the maximum length of  the sequences and the maximum vocabulary size which are essential Hyper parameters for the Transformer . 

**Encoder:**  The file of `layers.Transformer_layers.py` have the `Encoder` layer which takes the features extracted form the `Cnn model` to encode the information about the image content . 

**Decoder:** Which is also in `layers.Transformer_layers.py` have the `Decoder` layer where the magic happens the decoder learns the best way to map the caption to the image depending on its content .

**Image_caption:** The basic model of the transformer `Image_caption.py` that contains the basic function for loss and accuracy calculation .
**caption_Module** includes `Caption_module.py`  contains the caption_maker `Tensorflow Module` that uses the `tf.function decorator(@)` to speed up the process and create the graph which is essential for production .




