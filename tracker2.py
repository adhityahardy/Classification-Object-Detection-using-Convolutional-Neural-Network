import cv2
import math
import time
import numpy as np
# import pickle
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from tensorflow.keras import datasets, layers, models



import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="test",
    auth_plugin='mysql_native_password',
)
mycursor = db.cursor()


limit = 80  # km/hr

def create_model():
    cnn = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3),
                      activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return cnn


cnn = create_model()
checkpoint_path = "C:/Users/adhit/Downloads/Kampus Merdeka/Final/training_1/cp.ckpt"
cnn.load_weights(checkpoint_path)
# cnn = pickle.load(
#     open('C:/Users/adhit/Downloads/Kampus Merdeka/Final/trained_model.pickle', 'rb'))
classes = ["airplane", "Mobil", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "Truck"]

file = open(
    "C:/Users/adhit/Downloads/Kampus Merdeka/Final/SpeedRecord.txt", "w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        self.id_count = 0
        #self.start = 0
        #self.stop = 0
        self.et = 0
        self.s1 = np.zeros((1, 1000))
        self.s2 = np.zeros((1, 1000))
        self.s = np.zeros((1, 1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # CHECK IF OBJECT IS DETECTED ALREADY
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True

                    # START TIMER
                    if (y >= 410 and y <= 430):
                        self.s1[0, id] = time.time()

                    # STOP TIMER and FIND DIFFERENCE
                    if (y >= 235 and y <= 255):
                        self.s2[0, id] = time.time()
                        self.s[0, id] = self.s2[0, id] - self.s1[0, id]

                    # CAPTURE FLAG
                    if (y < 235):
                        self.f[id] = 1

            # NEW OBJECT DETECTION
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.s[0, self.id_count] = 0
                self.s1[0, self.id_count] = 0
                self.s2[0, self.id_count] = 0

        # ASSIGN NEW ID to OBJECT
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # SPEEED FUNCTION
    def getsp(self, id):
        if (self.s[0, id] != 0):
            s = 214.15 / self.s[0, id]
        else:
            s = 0

        return int(s)

    # SAVE VEHICLE DATA
    def capture(self, img, x, y, h, w, sp, id):
        if(self.capf[id] == 0):
            self.capf[id] = 1
            self.f[id] = 0
            crop_img = img[y-5:y + h+5, x-5:x + w+5]

            # img_det = mpimg.imread(crop_img)
            resized_image = resize(crop_img, (32, 32, 3))
            predictions = cnn.predict(np.array([resized_image]))
            x = predictions
            list_index = [0,1,2,3,4,5,6,7,8,9]
            for i in range(10):
                for j in range(10):
                    if x[0][list_index[i]] > x[0][list_index[j]]:
                        temp = list_index[i]
                        list_index[i] = list_index[j]
                        list_index[j] = temp
                    # print(list_index)
            # for l in list_index:
            #     if (l == (1 or 9)):
            #         test = l
            #         break

            # for n,l in enumerate(list_index):
            #     if(list_index[n]== (1 or 9)):
            #         test = l 
            #         break

            # prediksi = classes[test]
            y_classes = [np.argmax(element) for element in x]

            prediksi = classes[y_classes[0]]
            # if prediksi != ("Mobil" or "Truck"):
            #     prediksi = "Mobil"

            n = str(id)+"_speed_"+str(sp)+"_" +str(prediksi)
            file = 'C:/Users/adhit/Downloads/Kampus Merdeka/tutorial/static/uploads/kendaraan/' + n + '.jpg'

            cv2.imwrite(file, crop_img)
            self.count += 1
            filet = open(
                "C:/Users/adhit/Downloads/Kampus Merdeka/Final/SpeedRecord.txt", "a")
            a = round(predictions[0][list_index[0]]*100)
            if(sp > limit):
                file2 = 'C:/Users/adhit/Downloads/Kampus Merdeka/tutorial/static/uploads/pelanggar/' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(id)+" \t "+str(sp)+"<---exceeded\n")
                mycursor.execute(
                "INSERT INTO kendaraan_pelanggar (nama_gambar, kecepatan, jenis_kendaraan, akurasi) VALUES (%s, %s, %s, %s)", (n+ '.jpg', sp, prediksi, a))
                db.commit()
                self.exceeded += 1
            else:
                filet.write(str(id) + " \t " + str(sp) + "\n")
                mycursor.execute(
                "INSERT INTO kendaraan (nama_gambar, kecepatan, jenis_kendaraan, akurasi) VALUES (%s, %s, %s, %s)", (n+ '.jpg', sp, prediksi, a))
                db.commit()
            filet.close()

    # SPEED_LIMIT

    def limit(self):
        return limit

    # TEXT FILE SUMMARY
    def end(self):
        file = open("C:/Users/adhit/Downloads/Kampus Merdeka/Final/SpeedRecord.txt", "a")
        file.write("\n-------------\n")
        file.write("-------------\n")
        file.write("SUMMARY\n")
        file.write("-------------\n")
        file.write("Total Vehicles :\t"+str(self.count)+"\n")
        file.write("Exceeded speed limit :\t"+str(self.exceeded))
        file.close()
