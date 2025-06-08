from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask import jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model('model/reefident_final_model.keras')
class_names = ['invasive','noninvasive']  # Update if different

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 224, 224, 3)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            return redirect(request.url)
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            # Prediction
            img = prepare_image(filepath)
            
            prediction = model.predict(img)[0][0]
            print("Raw prediction:", prediction)  # Debug output
            if prediction >= 0.5:
                predicted_label = 'noninvasive'
                confidence = prediction * 100
            else:
                predicted_label = 'invasive'
                confidence = (1 - prediction) * 100


            return render_template('index.html', prediction=predicted_label, confidence=confidence, image_file=filename)

    return render_template('index.html', prediction=None)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    img = prepare_image(filepath)
    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        predicted_label = 'noninvasive'
        confidence = prediction * 100
    else:
        predicted_label = 'invasive'
        confidence = (1 - prediction) * 100

    return jsonify({
        'label': predicted_label,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', debug=True)

