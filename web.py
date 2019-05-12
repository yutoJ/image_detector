import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from PIL import Image
import numpy as np
from keras.models import Sequential, load_model
import keras, sys

classes = ["monkey", "crow", "boar"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

      model = load_model('./animal_cnn_aug.h5')

      image = Image.open(filepath)
      image = image.convert('RGB')
      image = image.resize((image_size, image_size))
      data = np.asarray(image)
      X = []
      X.append(data)
      X = np.array(X)

      result = model.predict([X])[0]
      predicted = result.argmax()
      percentage = int(result[predicted] * 100)

      return classes[predicted] + str(percentage) + " %"
      # return redirect(url_for('uploaded_file', filename=filename))
  return '''
  <!doctype html>
  <title>Upload new File</title>
  <h1>Upload new File</h1>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)