from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

sys.path.append("..")
from Project.utils import label_map_util

from Project.utils import visualization_utils as vis_util


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

prediction_task = ''

# from utils import visualization_utils as vis_util
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
###############################################################################

curr_dir = os.getcwd()
prototxtPath = os.path.join(curr_dir,'face_detector','deploy.prototxt')
model_name = os.path.join(curr_dir,'mask_detector.model')
weightsPath = os.path.join(curr_dir,'face_detector','res10_300x300_ssd_iter_140000.caffemodel')
###############################################################################

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
###############################################################################

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
###############################################################################

#home page
@app.route('/')
def index():
    return render_template('index.html')
###############################################################################

#redirecting to upload the image for testing based on taksks
@app.route('/upload',methods=['POST'])
def upload_image():
    prediction_task = request.form['predict_button']
    return render_template('upload_image.html',data=prediction_task)
###############################################################################

#saving the uploaded image to a folder
@app.route('/image_prediction',methods=['POST'])
def load_image():
    prediction_task = request.form['prediction_task']
    print(prediction_task)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        if prediction_task == 'Face':
            return redirect(url_for('face_mask_prediction',filename=filename))
        else:
            return redirect(url_for('human_prediction',filename=filename))
###############################################################################

#Face mask detection functions
@app.route('/face_mask_prediction/<filename>')
def face_mask_prediction(filename):

	image_path = app.config['UPLOAD_FOLDER']
	test_image = [os.path.join(image_path,filename.format(i))
                 for i in range(1,2)]

	net = cv2.dnn.readNet(prototxtPath,weightsPath)
	print("[INFO] loading face detector model...")

	model = load_model(model_name)
	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(test_image[0])
	orig = image.copy()
	(h, w) = image.shape[:2]

	# construct a blob from the image
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	cv2.imwrite('images/prediction.jpg', image)
	return send_from_directory(app.config['UPLOAD_FOLDER'],'prediction.jpg')
###############################################################################

#human prediction task using tensorflow
@app.route('/human_prediction/<filename>')
def human_prediction(filename):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                #print(image)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                indices = np.argwhere(classes == 1)
                boxes = np.squeeze(boxes[indices])
                scores = np.squeeze(scores[indices])
                classes = np.squeeze(classes[indices])

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                im.save('images/' + filename)

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
###############################################################################

