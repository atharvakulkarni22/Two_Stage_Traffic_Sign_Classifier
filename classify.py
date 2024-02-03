import numpy as np
import cv2
import time
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from keras.models import load_model

def classify(path):
    try:

        currentDirectory = os.getcwd()
        namesFilePath = os.path.join(currentDirectory, "final_models", "category.names")
        configFilePath = os.path.join(currentDirectory, "final_models", "yolov3.cfg")
        yoloWeightsPath = os.path.join(currentDirectory, "final_models", "yolov3.weights")
        cnnModelPath = os.path.join(currentDirectory, "final_models", "cnn26.h5")
        classesFilePath = os.path.join(currentDirectory, "final_models", "classes.txt")
        resultTextPath = os.path.join(currentDirectory, "result", "result_out.txt")
        resultImagePath = os.path.join(currentDirectory, "result", "result.jpg")

        # Reading image with OpenCV library
        # OpenCV by default reads images in BGR format
        print("\n\n")
        print("Image Path:", path)
        image_BGR = cv2.imread(path)
        
        # Showing image shape
        print('Image shape:', image_BGR.shape)

        h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements
        # Showing height an width of image
        print('Image height={0} and width={1}'.format(h, w))  # 466 700

        # Getting blob from input image
        # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Resulted shape has number of images, number of channels, width and height
        # print('Blob shape:', blob.shape)  # (1, 3, 416, 416)

        
        # Getting labels reading every line and putting them into the list labels
        with open(namesFilePath) as names:
            labels = [line.strip() for line in names]

        # Loading trained YOLO v3 Object Detector with the help of 'dnn' library from OpenCV
        network = cv2.dnn.readNetFromDarknet(configFilePath, yoloWeightsPath)

        # Getting list with names of all layers from YOLO v3 network
        allLayers = network.getLayerNames()
        # print(layers_names_all)

        # Getting only output layers' names that we need from YOLO v3 network
        outputLayers = [allLayers[i[0] - 1] for i in network.getUnconnectedOutLayers()]
        # print(outputLayers)

        # Setting minimum probability to eliminate weak predictions
        minimumProbability = 0.5
        
        # Setting threshold for filtering weak bounding boxes with non-maximum suppression
        threshold = 0.3

        # Generating colours for representing every detected object
        # with function randint(low, high=None, size=None, dtype='l')
        # colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        # print(len(colours))
        # colours = np.random.randint(255, 0, 0)
        # colours = (255,0,0)
        # setting blob as input to the network
        network.setInput(blob)  
        start = time.time()
        # Implementing forward pass with the input blob only through output layers
        networkOutput = network.forward(outputLayers)
        end = time.time()
        print('\nObjects Detection took {:.5f} seconds'.format(end - start))
        

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        boundingBoxes = []
        confidences = []
        classNumbers = []
        c1 = 0
        c2 = 0
        # Going through all output layers after feed forward pass
        # nO = [r0,r1,r2]
        # r0 = [d0,d1,d2,d3...]
        # d0 = [s0,s1,s2,s3...]
        # s0 = [0.index, 1.classId, 2.confidence, 3.x, 4.y, 5.width, 6.height]
        # nO = [r0[d0[s0[0,1,2,3,4,5,6],s1[0,1,2,3,4,5,6]...]...]...]
        for result in networkOutput:
            # Going through all detections from current output layer
            c1 += 1
            for detectedObjects in result:
                # Getting 4 classes' probabilities for current detected object
                scores = detectedObjects[5:]
                # Getting index of the class with the maximum value of probability
                currentClass = np.argmax(scores)
                # Getting value of probability for defined class
                currentProbability = scores[currentClass]
                c2 += 1
                # Eliminating weak predictions with minimum probability
                if currentProbability > minimumProbability:
                    
                    # Scaling bounding box coordinates to the size of original image
                    currentBox = detectedObjects[0:4] * np.array([w, h, w, h])
                    xCenter, yCenter, boxWidth, boxHeight = currentBox
                    # xCenter = int(detectedObjects[0] * w)
                    # yCenter = int(detectedObjects[1] * h)
                    # boxWidth = int(detectedObjects[2] * w)
                    # boxHeight = int(detectedObjects[3] * w)
                    

                    # Now, from YOLO data format, we can get top left corner coordinates 
                    # that are xMin and yMin
                    xMin = int(xCenter - (boxWidth / 2))
                    yMin = int(yCenter - (boxHeight / 2))

                    # Adding results into prepared lists
                    boundingBoxes.append([xMin, yMin, int(boxWidth), int(boxHeight)])
                    confidences.append(float(currentProbability))
                    classNumbers.append(currentClass)
        
        # print(c1,c2)
        # print("#bounding box: ",len(boundingBoxes))
        # Implementing non-maximum suppression of given bounding boxes
        # With this technique we exclude some of bounding boxes if their
        # corresponding confidences are low or there is another
        # bounding box for this region with higher confidence
        results = cv2.dnn.NMSBoxes(boundingBoxes, confidences, minimumProbability, threshold)
        print("Number of Signs Detected : %d" %len(results))

        # Checking if there is at least one detected object after non-maximum suppression
        counter = 1
        f = open(resultTextPath, "w")
        f.close()
        if len(results) > 0:
            model = load_model(cnnModelPath)
            # model = load_model('G:\\tsd\\classification\\cnn2\\models\\model13.h5')
            for i in range(len(boundingBoxes)):
                if i in results:
                    # Getting current bounding box coordinates, width and height
                    xMin, yMin, boxWidth, boxHeight = boundingBoxes[i]
                    # Preparing colour for current bounding box and converting from numpy array to list
                    # currentBoxColour = colours[classNumbers[i]].tolist()

                    # Drawing bounding box on the original image
                    cv2.rectangle(image_BGR, (xMin, yMin), (xMin + boxWidth, yMin + boxHeight),(0, 0, 255), 3)
                    
                    # Crop the image to feed it to the classifier
                    cropImage = image_BGR[yMin:yMin+boxHeight, xMin:xMin+boxWidth]

                    # preprocessing image for classification
                    # convert the cropped image to numpy array
                    cropImage = np.asarray(cropImage)
                    # resize the cropped image to 32, 32
                    cropImage = cv2.resize(cropImage, (32, 32))
                    # convert the image to grayscale
                    cropImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
                    # apply equalization 
                    cropImage = cv2.equalizeHist(cropImage)
                    # normalizing  the image data
                    cropImage = cropImage/255
                    # reshaping the image tupule to since now it has only one channel
                    cropImage = cropImage.reshape(1, 32, 32, 1)
                    
                    # predict the traffic sign and match the predicted class to correct text from sign_classes.txt
                    with open(classesFilePath, "r") as f:
                        classes = [line.strip() for line in f.readlines()]
                        
                        classProb = model.predict(cropImage)
                        prediction = np.argmax(classProb)
                        prob = float(classProb[0][prediction])
                        label = '%s. %s' % (counter, classes[prediction])
                        #put label
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        top = max(yMin, labelSize[1])
                        cv2.rectangle(image_BGR, (xMin, top - round(1.5*labelSize[1])), (xMin + 
                        round(labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)

                        cv2.putText(image_BGR, label, (xMin, yMin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                        # prediction = np.argmax(model.predict(cropImage))
                        # # prediction_label = str(counter)+": "+str(classes[prediction])
                        # prediction_label = str(counter)+": "+str(prediction)


                    f.close
                    # put text on the original image using the prediction label
                    # cv2.putText(image_BGR, prediction_label, (xMin-10, yMin - 5),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, currentBoxColour, 2)

                    with open(resultTextPath,"a") as f:
                        # output = str(counter)+". "+str(prediction)+"- "+str(classes[prediction])+" Confidence:%0.2f" % classProb[prediction]+"\n"
                        output = str(counter)+". "+str(prediction)+"- "+str(classes[prediction])+": %0.3f" %(prob*100)+"%\n"
                        f.write(output)

                    print(str(counter)+". "+str(prediction)+"- "+str(classes[prediction])+": %0.3f" %(prob*100)+"%")
                    counter +=1
        # print("counter: ",counter)
        #store the result image
        cv2.imwrite(resultImagePath, image_BGR)
    except:
        with open(resultTextPath,"w") as f:
            f.write("Error in reading image, please try again")
