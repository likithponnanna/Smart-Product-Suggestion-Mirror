from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

from cv2 import *

from dataGet import suggestProduct
from getAllAttr import convertAges, getUserId, getCityName, getPriceForName

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(2, 6)', '(6, 14)', '(15, 19)', '(20, 31)', '(32, 40)', '(44, 55)', '(56, 100)']
user_ids = [1000006,1000009,1000013,1000134]
gender_list = ['Male', 'Female']


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
        "age_gender_model/deploy_age.prototxt",
        "age_gender_model/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "age_gender_model/deploy_gender.prototxt",
        "age_gender_model/gender_net.caffemodel")

    return (age_net, gender_net)


def capture_image(width, height):
    font = cv2.FONT_HERSHEY_TRIPLEX
    s, frame = cam.read()

    age = 10
    gender = 'Male'
    x = 0
    y = height
    face_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml ')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # Draw a rectangle around every found face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (66, 244, 89), 2)
        face_img = frame[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        age = convertAges(age)
        overlay_text = "%s, %s" % (gender, age)
       # overlay_text1 = "%s" % (suggestProduct(2)[0])
       # overlay_text2 = "Buy: %s" % (suggestProduct(2)[1])
       # cv2.putText(frame, overlay_text1, (0, int(height - 20)), font, 0.8, (500, 200, 200), 2, cv2.LINE_AA)
       # cv2.putText(frame, overlay_text2, (x, y - 40), font, 0.8, (500, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, overlay_text, (x, y), font, 0.8, (196, 231, 237), 1, cv2.LINE_AA)

    cv2.imshow("Image", frame)

    imwrite("/tmp/image.jpg", frame)
    destroyWindow("cam-test")
    return age, gender,frame,x,y,font,faces



def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label



if __name__ == "__main__":
    age_net, gender_net = initialize_caffe_model()
    loaded_model = joblib.load("product_recomm_model/main_model/Random_Forest_Prod.pkl")
    Encoder = joblib.load("product_recomm_model/main_model/LabelEncode_Product.pkl")
    Scaler = joblib.load("product_recomm_model/main_model/Scaler_Product.pkl")
    LDA = joblib.load("product_recomm_model/main_model/LDA_Product.pkl")
    Age_loaded_model = joblib.load("product_recomm_model/age_gender_prod_model/Age Gender Prod recommend models/Age_Random_Forest.pkl")
    Age_Encoder = joblib.load("product_recomm_model/age_gender_prod_model/Age Gender Prod recommend models/Age_LabelEncode.pkl")
    Age_Scaler = joblib.load("product_recomm_model/age_gender_prod_model/Age Gender Prod recommend models/Age_Scaler.pkl")
    Age_LDA = joblib.load("product_recomm_model/age_gender_prod_model/Age Gender Prod recommend models/Age_LDA.pkl")
    cam = VideoCapture(0)
    #download_dir = "newPeople.csv"
    #csv = open(download_dir, "a")
    #columnTitleRow = "name, age, gender\n"
    #csv.write(columnTitleRow)
    #cv2.resize(cam, (1000, 1000))
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # new addition
    #time.sleep(0.1)
    #lookingCount = 1;
    age, gender, frame, x, y, font, faces = capture_image(width, height)
    while (True):
        #capture_image(width, height)
        file_name = "/tmp/image.jpg"
        model_file = "tf_files/retrained_graph.pb"
        label_file = "tf_files/retrained_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = 'Mul'
        output_layer = "final_result"
        parser = argparse.ArgumentParser()
        parser.add_argument("--image", help="image to be processed")
        parser.add_argument("--graph", help="graph/model to be executed")
        parser.add_argument("--labels", help="name of file containing labels")
        parser.add_argument("--input_height", type=int, help="input height")
        parser.add_argument("--input_width", type=int, help="input width")
        parser.add_argument("--input_mean", type=int, help="input mean")
        parser.add_argument("--input_std", type=int, help="input std")
        parser.add_argument("--input_layer", help="name of input layer")
        parser.add_argument("--output_layer", help="name of output layer")
        args = parser.parse_args()

        if args.graph:
            model_file = args.graph
        if args.image:
            file_name = args.image
        if args.labels:
            label_file = args.labels
        if args.input_height:
            input_height = args.input_height
        if args.input_width:
            input_width = args.input_width
        if args.input_mean:
            input_mean = args.input_mean
        if args.input_std:
            input_std = args.input_std
        if args.input_layer:
            input_layer = args.input_layer
        if args.output_layer:
            output_layer = args.output_layer

        graph = load_graph(model_file)
        t = read_tensor_from_image_file(file_name,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end = time.time()
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        k = 0
        temp = 0
        print("faces",faces)
        for i in results[:]:
            i = results[k]
            i *= 100
            if i >= 80:
                temp = -1
                print('Recognised as', labels[k])
                userId = getUserId(labels[k])
                #print("is it true", labels[k].lower() is not 'not anyone' )
                if faces is not () and labels[k].lower() == 'not anyone':
                    overlay_text2 = "Hello Patron, Please look around and consider our tailored suggestions to you"
                    cv2.putText(frame, overlay_text2, (10, 70), font, 0.5, (66, 244, 209), 1, cv2.LINE_AA)
                    cv2.imshow("Image", frame)
                    destroyWindow("cam-test")
                elif (str(age) is not '10') and (labels[k].lower() is not "not anyone") and (faces is not ()):
                    predictionArr = [[userId, gender, age, getCityName(), getPriceForName(labels[k])]]
                    predictionArr = np.array(predictionArr)
                    # print("pred arr", predictionArr)
                    predictionArr[:, 1] = Encoder.fit_transform(predictionArr[:, 1])
                    predictionArr[:, 2] = Encoder.fit_transform(predictionArr[:, 2])
                    predictionArr[:, 3] = Encoder.fit_transform(predictionArr[:, 3])
                    predictionArr = Scaler.transform(predictionArr)
                    predictionArr = LDA.transform(predictionArr)
                    predicted_cat = loaded_model.predict(predictionArr)
                    overlay_text2 = "Hi %s, Buy: %s in %s" % (
                    labels[k], (suggestProduct(predicted_cat)[1]), (suggestProduct(predicted_cat)[0]))
                    cv2.putText(frame, overlay_text2, (10, 60), font, 0.5, (66, 244, 209), 1, cv2.LINE_AA)
                    cv2.imshow("Image", frame)
                    destroyWindow("cam-test")
                print(labels[k], round(results[k] * 100, 2), "%")
            elif temp == -1 or temp == 0:
               temp = 1
               print('Looking for faces!')
               if faces is not () and (labels[k].lower() != 'not anyone'):
                    predictionArr = [[random.choice(user_ids), gender, age, getCityName()]]
                    predictionArr = np.array(predictionArr)
                    predictionArr[:, 1] = Age_Encoder.fit_transform(predictionArr[:, 1])
                    predictionArr[:, 2] = Age_Encoder.fit_transform(predictionArr[:, 2])
                    predictionArr[:, 3] = Age_Encoder.fit_transform(predictionArr[:, 3])
                    predictionArr = Age_Scaler.transform(predictionArr)
                    predictionArr = Age_LDA.transform(predictionArr)
                    predicted_cat = Age_loaded_model.predict(predictionArr)
                    overlay_text2 = "Hello Patron, Please look around and consider our tailored suggestions to you"
                    cv2.putText(frame, overlay_text2, (10, 30), font, 0.5, (66, 244, 209), 1, cv2.LINE_AA)
                    overlay_text2 = "Buy: %s in %s" % (suggestProduct(predicted_cat)[1],suggestProduct(predicted_cat)[0])
                    overlay_text3 = "*Suggestions based on demographic information"
                    cv2.putText(frame, overlay_text2, (150, int(height)-20), font, 0.5, (93, 206, 78), 1, cv2.LINE_AA)
                    cv2.putText(frame, overlay_text3, (170, int(height)-10), font, 0.3, (93, 206, 78), 1, cv2.LINE_AA)
                    cv2.imshow("Image", frame)
                    destroyWindow("cam-test")
               print(labels[k], round(results[k] * 100, 2), "%")
            k += 1
        print("\n")
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        age, gender,frame,x,y,font,faces = capture_image(width, height)
