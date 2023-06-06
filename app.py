#import the necessary models and libraries.
from __future__ import division, print_function
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import sys
import os
# Flask utilities
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

#used a pre trained model
#import the model, image processor and tokenizer from the pretrained model.
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

      

def predict_caption(image_path, num_caps): #function to predict caption which accepts image path& no. of captions you want.
  images = []
  img = Image.open(image_path)
  if img.mode != "RGB":
    img = img.convert(mode="RGB")

    images.append(img)
#Encoding
  pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values #generate pixel values using feature extractor
  pixel_values = pixel_values.to(device) #store the pixel values to device.
  max_length =  16 #maximum length of a caption.
  num_beams =  10  #number of beams,(its a parameter of the model.)
  num_caps= int(num_caps) 
  gen_kwargs = {"max_length":max_length,"num_beams": num_beams , "num_return_sequences": num_caps}
#Decoding 
  output_ids = model.generate(pixel_values, **gen_kwargs) #generate output ids using pixel values.
  predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True) #decode the output ids and tokenize them.
  predictions = [predictions.strip() for predictions in predictions] #remove the spaces at the end or beginning of a string
  return predictions


  
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file'] #file provided by the user.
        caps = request.form.get('number') #extract the number of captions provided by the user.
        

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Call the predict_caption function to predict the captions.
        captions = predict_caption(file_path, caps)
        result= captions
        return result
    return None
if __name__ == '__main__':
   app.run(debug=True)