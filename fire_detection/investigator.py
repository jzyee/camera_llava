import sys
import cv2
from PyQt5.QtWidgets import QScrollArea, QMessageBox, QCheckBox, QSizePolicy, QApplication, QWidget,  QHBoxLayout, QVBoxLayout,QListWidget, QStackedWidget, QPushButton, QLabel, QGridLayout, QTextEdit, QLineEdit, QRadioButton, QButtonGroup, QFileDialog
from PyQt5.QtCore import QRunnable, QObject, QThreadPool, QTimer, QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from llama_cpp import Llama
from openai import OpenAI
import base64
import numpy as np
import queue
import time
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms import functional as F
from transformers import pipeline


# Define the first layout in its own class
class PredictWorker(QThread):
    # Define a signal to send the result back to the main thread
    resultReady = pyqtSignal(str)

    def __init__(self, userInput, model, img):
        super().__init__()
        self.userInput = userInput
        self.img = img
        self.processed_prompt = self.process_prompt()
        self.model = model
        

    def process_img(self):
            # encoded_string = base64.b64encode(self.img).decode('utf-8')
            # return f"data:image/jpeg;base64,{encoded_string}"
       
            with open("temp.jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded_string}"
    
    def process_prompt(self):
        
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.process_img(),
                        },
                    },
                    {"type": "text", "text": f"{self.userInput}"},
                ],
            }
        ]
        return messages
        


    def run(self):
        # Placeholder for a long-running prediction model
        # Simulate a delay to mimic a time-consuming task
        response = self.model.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=self.processed_prompt
        )
        # print(output)
        # result = output
        result = f"'{response.choices[0].message.content}'"
         # Simulate a long task
        self.resultReady.emit(result)


class AlertWorker(QThread):
    caption_signal = pyqtSignal(str, float)  # Signal to emit the caption
    
    def __init__(self, userInput, model, img):
        super().__init__()
        self.framesQueue = queue.Queue()
        self.running=True
        self.img = img
        self.userInput = userInput
        # self.processed_prompt = self.process_prompt()
        self.model = model
        self.timeOfSampling = time.time()
        self.frame = None

    # @pyqtSlot(np.ndarray)
    def add_frame(self, frame):
        timestamp = time.time()
        self.framesQueue.put((frame, timestamp))
        if not self.isRunning():
            self.start()

    def stop(self):
        self.running = False
        self.wait()
    
    def process_img(self):
            # encoded_string = base64.b64encode(self.img).decode('utf-8')
            # return f"data:image/jpeg;base64,{encoded_string}"
            retval, buffer = cv2.imencode('.jpg', self.frame)

            if retval:
                jpg_as_text = base64.b64encode(buffer)
                jpg_as_str = jpg_as_text.decode('utf-8')
                return f"data:image/jpeg;base64,{jpg_as_str}"
            
            return None
    
    def process_prompt(self):
        
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.process_img(),
                        },
                    },
                    {"type": "text", "text": f"{self.userInput}"},
                ],
            }
        ]
        return messages
        


    def run(self):
        while self.running:
            if not self.framesQueue.empty():
                # Placeholder for a long-running prediction model
                # Simulate a delay to mimic a time-consuming task
                frame, timestamp = self.framesQueue.get()
                self.frame = frame
                self.processed_prompt = self.process_prompt()
                response = self.model.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=self.processed_prompt,
                    max_tokens=30,
                    temperature=0.4,
                    top_p=0.4
                )
                # print(output)
                # result = output
                caption = f"'{response.choices[0].message.content}'"
                # Simulate a long task
                self.caption_signal.emit(caption, timestamp)
    

class AlertSystem(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initVideoStream()
        self.model = self.initModel()
        self.initPredictWorker()
        self.alertSystemEnabled = False
        self.triggerWords = {"knife", "dangerous", "smoking"}
        self.lastProcessedTime = time.time()
        self.samplingInterval = 10

        

    def initModel(self):
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-1234")
        return client

    def initUI(self):
        self.gridLayout = QVBoxLayout()


        # alert banner
        self.bannerLabel = QLabel()
        self.bannerLabel.setText("Alert System Banner")
        self.bannerLabel.setAlignment(Qt.AlignCenter)
        # Set the initial background color to green and text color to white
        self.bannerLabel.setStyleSheet("background-color: green; color: white;")
        self.bannerLabel.setFixedHeight(30)


        # init the video portion
        self.videoLabel = QLabel()
        self.videoLabel.setScaledContents(True)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # init the alert window
        self.alertWindow = QTextEdit()
        self.alertWindow.setReadOnly(True)
        self.alertWindow.setMaximumHeight(30)
        self.alertWindow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # init toggle switch
        self.toggleSwitch = QCheckBox("Enable Alert System")
        self.toggleSwitch.stateChanged.connect(self.toggleAlertSystem)

        # adding to the overall layout
        self.gridLayout.addWidget(self.bannerLabel)
        self.gridLayout.addWidget(self.toggleSwitch)
        self.gridLayout.addWidget(self.alertWindow)
        self.gridLayout.addWidget(self.videoLabel)
        
        

        self.setLayout(self.gridLayout)

        self.prompt = "Assume the role of a police officer. Be factual and precise. is there person in the center carrying a sharp object in his or her hand."

    def initVideoStream(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)

    def initPredictWorker(self):
        self.predictWorker = AlertWorker(self.prompt, self.model, self.displayImage)
        self.predictWorker.caption_signal.connect(self.updateCaption)

    def toggleAlertSystem(self, state):
        self.alertSystemEnabled = bool(state)
        if not self.alertSystemEnabled:
            self.alertWindow.clear()

    def updateFrame(self):
        ret, frame = self.cap.read()
        currentTime = time.time()
        if ret:
            
            if self.alertSystemEnabled:
                self.displayImage(frame)
                if currentTime - self.lastProcessedTime >= self.samplingInterval:
                    self.predictWorker.add_frame(frame)
                    self.lastProcessedTime = currentTime


    def updateCaption(self, caption, timestamp):
        if self.alertSystemEnabled: #and any(triggerWord in caption for triggerWord in self.triggerWords):
            # self.alertWindow.setText(caption)
            # Process the caption and timestamp
            # Check if the caption contains the word "yes"
            if "yes" in caption.lower():
                # Change the banner's background color to red
                self.bannerLabel.setStyleSheet("background-color: red; color: white;")
            else:
                # Revert the banner's background color to green
                self.bannerLabel.setStyleSheet("background-color: green; color: white;")

            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            display_text = f"{formatted_time}: {caption}"
            self.alertWindow.setText(display_text)

    def displayImage(self, img):
        aspect_ratio = img.shape[1] / img.shape[0]
        width = self.videoLabel.width()
        height = int(width / aspect_ratio)
        img = cv2.resize(img, (width, height))
        qformat = QImage.Format_RGB888
        outImage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat).rgbSwapped()
        self.videoLabel.setPixmap(QPixmap.fromImage(outImage))

    # def displayImage(self, img, label):
    #     img = cv2.resize(img, (350, 250))
    #     qformat = QImage.Format_RGB888
    #     outImage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat).rgbSwapped()
    #     label.setPixmap(QPixmap.fromImage(outImage))
        
    def confirmStreamSource(self):
        
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if fileName:
            self.useFile(fileName)

    def useWebcam(self):
        self.changeStreamSource(0)

    def useFile(self, filePath):
        self.changeStreamSource(filePath)

    def changeStreamSource(self, source):
        if self.cap:
            self.cap.release()
            self.timer.stop()
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.timer.start(20)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.predictWorker.isRunning():
                self.predictWorker.stop()  # Stop the PredictWorker thread
            
            if self.cap.isOpened():
                self.cap.release()  # Release the video capture resource
            
            event.accept()
            print("Window closed cleanly")
        else:
            event.ignore()




class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initVideoStream()
        self.model = self.initModel()
        self.evalFrame = None

    def initModel(self):
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-1234")
        return client

    def initUI(self):
        # Initialize grid layout
        self.gridLayout = QGridLayout()

        

        # Video and snapshot labels
        self.videoLabel = QLabel()
        self.snapshotLabel = QLabel()
        self.videoLabel.setScaledContents(True)
        self.snapshotLabel.setScaledContents(True)

        # Snapshot button
        self.snapshotButton = QPushButton("Take Snapshot")
        self.snapshotButton.clicked.connect(self.takeSnapshot)

        # Text window and input field for predictions
        self.textWindow = QTextEdit()
        self.textWindow.setReadOnly(True)
        self.inputField = QLineEdit()

        # Predict button
        self.predictButton = QPushButton("Predict")
        self.predictButton.clicked.connect(self.sendToModel)

        # Radio buttons for stream source selection
        self.webcamRadio = QRadioButton("Webcam Stream")
        self.webcamRadio.setChecked(True)
        self.fileRadio = QRadioButton("Uploaded File Stream")
        self.streamConfigButton = QPushButton("Confirm Stream Source")
        self.streamConfigButton.clicked.connect(self.confirmStreamSource)

        # Add buttons to right sidebar grid

        self.right_sidebar_layout = QGridLayout()

        # Add media type buttons to right sidebar grid

        self.media_layout = QGridLayout()
        self.media_layout.addWidget(self.webcamRadio, 0, 0)
        self.media_layout.addWidget(self.fileRadio, 0, 1)
        self.media_layout.addWidget(self.streamConfigButton, 1, 0)

        self.right_sidebar_layout.addLayout(self.media_layout, 0,0,2,3)
        self.right_sidebar_layout.addWidget(self.snapshotButton, 1, 0, 1, 3)
        self.right_sidebar_layout.addWidget(self.textWindow, 2, 0, 3, 3)
        self.right_sidebar_layout.addWidget(self.inputField, 6, 0, 1, 1)
        self.right_sidebar_layout.addWidget(self.predictButton, 6, 2, 1, 1)

        # Adding widgets to layout
        self.gridLayout.addWidget(self.videoLabel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.snapshotLabel, 1, 0, 1, 1)
        self.gridLayout.addLayout(self.right_sidebar_layout, 0, 1, 3, 1)
        
        
        
        

        self.setLayout(self.gridLayout)

    def initVideoStream(self):
        # Initialize webcam as default video stream
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.displayImage(frame, self.videoLabel)

    def displayImage(self, img, label):
        img = cv2.resize(img, (350, 250))
        qformat = QImage.Format_RGB888
        outImage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(outImage))

    def takeSnapshot(self):
        ret, frame = self.cap.read()
        if ret:
            self.displayImage(frame, self.snapshotLabel)
            self.evalFrame = frame
            cv2.imwrite("temp.jpg",frame)

    def sendToModel(self):
        userInput = self.inputField.text()
        # Start the worker thread for the prediction model
        self.worker = PredictWorker(userInput, self.model, self.evalFrame)
        self.worker.resultReady.connect(self.onModelPrediction)
        self.worker.start()

    def onModelPrediction(self, modelOutput):
        # Update UI with model output
        userInput = self.inputField.text()
        self.textWindow.append(f"User: {userInput}\nModel: {modelOutput}\n")
        self.inputField.clear()

    def confirmStreamSource(self):
        if self.fileRadio.isChecked():
            fileName, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
            if fileName:
                self.useFile(fileName)
        else:
            self.useWebcam()

    def useWebcam(self):
        self.changeStreamSource(0)

    def useFile(self, filePath):
        self.changeStreamSource(filePath)

    def changeStreamSource(self, source):
        if self.cap:
            self.cap.release()
            self.timer.stop()
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.timer.start(20)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()

class SearchProcessorTask(QRunnable):
    # this function is to process a frame concurrently

    def __init__(self, frame, pipe, prompt, i, fps, resultEmitter):
        super().__init__()
        self.frame = frame
        self.pipe = pipe
        self.prompt = prompt
        self.i = i
        self.resultEmitter = resultEmitter
        self.fps = fps

    def search_frame(self, frame, pipe, prompt):
        found = False
        proc_prompt = self.process_prompt()
        outputs = pipe(frame, 
                      prompt=proc_prompt, 
                      generate_kwargs={"max_new_tokens": 5})
        
        if len(outputs[0]['generated_text']) > 0:
            model_response = outputs[0]['generated_text'].split('ASSISTANT:')[1]
            if 'yes' in model_response.lower():
                found =  True
        
        return found


    def process_prompt(self):

        return f"USER: <image>\n{self.prompt} Reply with only yes or no\nASSISTANT:"

    def run(self):
        
        timestamp = self.i / self.fps
        frame_to_pil = F.to_pil_image(self.frame.permute(2,0,1)) if self.frame.dim() == 3 else F.to_pil_image(self.frame)

        pred = self.search_frame(frame_to_pil, self.pipe, self.prompt)

        processed_results = {
            "timestamp": timestamp,
            "frame": frame_to_pil,
            "pred": pred
        }
        print(processed_results)
        self.resultEmitter.resultReady.emit(processed_results)

class ResultEmitter(QObject):
    resultReady = pyqtSignal(object)


# Define the second layout in its own class
class SearchVid(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initVideoStream()
        # self.model = self.initModel()
        self.evalFrame = None
        # init the list to store the frames
        self.results = [] 
        self.interval_seconds = 1
        num_workers = 4
        self.threadPool = QThreadPool()
        self.threadPool.setMaxThreadCount(4) 
        self.initModel()

    def initModel(self):
        # init pipeline
        model_id = "./tinyllava-v1.0-1.1b-hf"
        self.pipe = pipeline("image-to-text", model=model_id)

    def initUI(self):
        # Initialize grid layout
        self.overallLayout = QVBoxLayout()

        # Video
        self.videoLayout = QVBoxLayout()

        # self.videoLayout.addStretch(1)
        # init the video portion
        self.videoLabel = QLabel()
        self.videoLabel.setScaledContents(True)
        self.videoLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.videoLayout.addWidget(self.videoLabel)
        self.videoLayout.setAlignment(self.videoLabel, Qt.AlignHCenter)

        # Button for 
        self.streamConfigButton = QPushButton("File Upload")
        self.streamConfigButton.clicked.connect(self.confirmStreamSource)

        # Add media type buttons to right sidebar grid

        self.media_layout = QGridLayout()
        self.media_layout.addWidget(self.streamConfigButton, 0, 0)


        # Add a prompt text field and button
        self.promptLayout = QGridLayout()
        # Add a text field for the connect button
        self.searchInput = QLineEdit()
        # Add a button to confirm the text field input
        self.startSearchButton = QPushButton("Search")
        self.startSearchButton.clicked.connect(self.startVideoSearch)
        self.promptLayout.addWidget(self.searchInput, 0,0,0,3)
        self.promptLayout.addWidget(self.startSearchButton, 0,3)


        # scroll area setup
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()



        # Add a horizontal layout place holder for the found images
        self.foundImagesLayout = QHBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        




        self.resultLabel = QLabel()

        # Adding widgets to layout
        self.overallLayout.addLayout(self.videoLayout)
        self.overallLayout.addLayout(self.media_layout)
        self.overallLayout.addLayout(self.promptLayout)
        self.overallLayout.addWidget(self.resultLabel)
        self.overallLayout.addWidget(self.scrollArea)
        
        self.setLayout(self.overallLayout)

    def clearFoundImagesLayout(self):
        while self.foundImagesLayout.count():
            item = self.foundImagesLayout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def startVideoSearch(self):
        # clearing before performing another search
        self.clearFoundImagesLayout()

        self.prompt = self.searchInput.text()
        self.resultLabel.setText("Video being processed")
        self.video, audio, info = read_video(self.filePath, pts_unit='sec')
        self.fps = info["video_fps"]
        self.frames_to_skip = int(self.fps * self.interval_seconds)


        self.resultEmitter = ResultEmitter()
        self.resultEmitter.resultReady.connect(self.handleResult)

        self.num_exp_results = len([i for i in range(0, self.video.size(0), self.frames_to_skip)])
        
        for i in range(0, self.video.size(0), self.frames_to_skip):
            
            task = SearchProcessorTask(self.video[i], self.pipe, self.prompt, i, self.fps, self.resultEmitter)
            self.threadPool.start(task)

    def handleResult(self, result):
        self.results.append(result)

        if len(self.results) == self.num_exp_results:
            self.finalizeResults()


    def finalizeResults(self):
        self.results.sort(key=lambda x: x['timestamp'])
        
        count = 0

        for result in self.results:
            if result['pred']:
                count += 1
                self.addImageToImageLayout(result['frame'], result['timestamp'])

        self.resultLabel.setText(f"Processed {len(self.results)} frames. {count} positive result(s) shown.")

        # need to clear after using
        self.results = []
        self.threadPool.waitForDone()

    def addImageToImageLayout(self, image, timestamp):
        pilImage = image.convert("RGBA")

        original_size = pilImage.size
        scaled_size = (original_size[0] // 10, original_size[1] // 10)
        pilImage = pilImage.resize(scaled_size, Image.Resampling.LANCZOS)

        data = pilImage.tobytes("raw", "BGRA")

        qim = QImage(data, 
                     pilImage.size[0], 
                     pilImage.size[1], 
                     QImage.Format_ARGB32
                    )
        pixmap = QPixmap.fromImage(qim)

        imageLabel = QLabel(self.scrollAreaWidgetContents)
        imageLabel.setPixmap(pixmap)

        # Create a label for the timestamp
        timestampLabel = QLabel(self.scrollAreaWidgetContents)
        timestampLabel.setText(f"Time: {timestamp:.2f}s")
        # timestampLabel.setAlignment(Qt.AlignCenter)

        # Create a vertical layout to stack the image and timestamp
        layout = QVBoxLayout()
        layout.addWidget(imageLabel)
        layout.addWidget(timestampLabel)

        # Create a container widget to hold the vertical layout
        container = QWidget(self.scrollAreaWidgetContents)
        container.setLayout(layout)

        # Add the container to the horizontal layout
        self.foundImagesLayout.addWidget(container)

        # Ensure the newly added image is visible
        self.scrollArea.ensureWidgetVisible(container)
        
    def initVideoStream(self):
        # Initialize webcam as default video stream
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.displayImage(frame, self.videoLabel)

    def displayImage(self, img, label):
        img = cv2.resize(img, (350, 250))
        qformat = QImage.Format_RGB888
        outImage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(outImage))
        
    def confirmStreamSource(self):
        
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if fileName:
            self.useFile(fileName)
            self.filePath = fileName

    def useWebcam(self):
        self.changeStreamSource(0)

    def useFile(self, filePath):
        self.changeStreamSource(filePath)

    def changeStreamSource(self, source):
        if self.cap:
            self.cap.release()
            self.timer.stop()
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.timer.start(20)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()

    
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Navigation Example")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        mainLayout = QHBoxLayout(self)

        # Navigation list
        self.navList = QListWidget()
        self.navList.insertItem(0, "Live")
        self.navList.insertItem(1, "Search")
        self.navList.insertItem(2, "QA")
        

        # Stacked widget for content
        self.stackedWidget = QStackedWidget()

        # Add layouts to stacked widget
        self.layoutOne = AlertSystem()
        self.layoutTwo = SearchVid()
        self.layoytThree = VideoPlayer()
        self.stackedWidget.addWidget(self.layoutOne)
        self.stackedWidget.addWidget(self.layoutTwo)
        self.stackedWidget.addWidget(self.layoytThree)

        # Add widgets to the main layout
        mainLayout.addWidget(self.navList)
        mainLayout.addWidget(self.stackedWidget)

        # Adjust the relative sizes
        # Set the navigation list to take up 1/12 of the space
        # and the content area to take up the remaining space
        mainLayout.setStretchFactor(self.navList, 1)
        mainLayout.setStretchFactor(self.stackedWidget, 11)

        # Connect navigation changes to display function
        self.navList.currentRowChanged.connect(self.display)

    def display(self, index):
        # Change the displayed widget
        self.stackedWidget.setCurrentIndex(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
