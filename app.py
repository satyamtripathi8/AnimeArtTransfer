from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from net import generator
from tools.utils import load_test_data, save_images
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Initialize TensorFlow session and model
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
test_real = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='test')

with tf.compat.v1.variable_scope("G_MODEL", reuse=False):
    test_generated = generator.G_net(test_real).fake

saver = tf.compat.v1.train.Saver()

# Load the pre-trained model
checkpoint_dir = 'AnimeGANv2/checkpoint/generator_Hayao_weight'
ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
else:
    print(" [*] Failed to find a checkpoint")
    raise Exception("Checkpoint not found!")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Process the image
        sample_image = np.asarray(load_test_data(input_path, [256, 256]))
        output_filename = f'ghibli_{filename}'
        output_path = os.path.join('results', output_filename)
        
        # Generate the anime-style image
        fake_img = sess.run(test_generated, feed_dict={test_real: sample_image})
        save_images(fake_img, output_path, input_path)
        
        return send_file(output_path, as_attachment=True)
    
    return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(debug=True) 