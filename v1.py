import sys,time,os,random
import urllib.request
from tkinter import *
import tkinter as tk
from tkinter.filedialog import *
from tkinter import messagebox as tkinmsgbox
try:
    from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QProgressBar
    from PyQt6.QtCore import Qt
    import pyqrcode
    import png
    from PIL import ImageTk, Image
    from PIL import ImageGrab
    import cv2
    import PIL.Image
    from pyqrcode import QRCode
    import moviepy
    import moviepy.editor
    import matplotlib.pyplot as plt
    import numpy as np
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtWebEngineWidgets import *
    from PyQt5.QtPrintSupport import *
except ModuleNotFoundError as e:
    a=str(e)
    if a=="ModuleNotFoundError: No module named 'PyQt6'":
        tkinmsgbox.showerror("Detected that PyQt6 is not installed!", "System has Detected that PyQt6 is not installed!click ok or close the window to install PyQt6!")
        os.system('pip3 install pyqt6')
        sys.exit(303)
    elif a=="ModuleNotFoundError: No module named 'pyqrcode'":
        tkinmsgbox.showerror("Detected that pyqrcode is not installed!", "System has Detected that pyqrcode is not installed!click ok or close the window to install pyqrcode!")
        os.system('pip3 install pyqrcode')
        sys.exit(404)
    elif a=="ModuleNotFoundError: No module named 'png'":
        tkinmsgbox.showerror("Detected that pypng is not installed!", "System has Detected that png is not installed!click ok or close the window to install pypng!")
        os.system('pip3 install pypng')
        sys.exit(404)
    elif a=="ModuleNotFoundError: No module named 'moviepy'":
        tkinmsgbox.showerror("Detected that moviepy is not installed!", "System has Detected that moviepy is not installed!click ok or close the window to install moviepy!")
        os.system('pip3 install moviepy')
        sys.exit(404)
    elif a=="ModuleNotFoundError: No module named 'matplotlib'":
        tkinmsgbox.showerror("Detected that matplotlib is not installed!", "System has Detected that matplotlib is not installed!click ok or close the window to install matplotlib!")
        os.system('pip3 install matplotlib')
        sys.exit(404)
    elif a=="ModuleNotFoundError: No module named 'moviepy'":
        tkinmsgbox.showerror("Detected that numpy is not installed!", "System has Detected that numpy is not installed!click ok or close the window to install numpy!")
        os.system('pip3 install numpy')
        sys.exit(404)
    elif a=="ModuleNotFoundError: No module named 'PyQt5'":
        tkinmsgbox.showerror("Detected that PyQt5 is not installed!", "System has Detected that PyQt5 is not installed!click ok or close the window to install PyQt5!")
        os.system('pip3 install PyQt5')
        sys.exit(404)
    #opencv-python
    elif a=="ModuleNotFoundError: No module named 'cv2'":
        tkinmsgbox.showerror("Detected that opencv-python is not installed!", "System has Detected that opencv-python is not installed!click ok or close the window to install opencv-python!")
        os.system('pip3 install opencv-python')
        sys.exit(404)
def screenrecordergui():
    screen_recorder  = Tk()
    screen_recorder.geometry("340x220")
    screen_recorder.title("Screen Recorder")
     # Show image using label
    label1 = Label( screen_recorder, image = None, bd=0)
    label1.pack()
     #Create and place the components
    title_label = Label(screen_recorder, text="Screen Recorder", font=("Ubuntu Mono", 16), bg="#02b9e5")
    title_label.place(relx=0.5,rely=0.1, anchor=CENTER)
    info_label = Label(screen_recorder, text="Enter 'e' to exit screen recording", bg="#02b9e5")
    info_label.place(relx=0.5,rely=0.3, anchor=CENTER)
    screen_button = Button(screen_recorder, text="Record Screen", command=screenrecorder, relief= RAISED)
    screen_button.place(relx=0.5,rely=0.6, anchor=CENTER)
     
    screen_recorder.mainloop()
def screenrecorder():
    #Obtain image dimensions 
    #Screen capture 
    image = ImageGrab.grab()
    #Convert the object to numpy array
    img_np_arr = np.array(image)
    #Extract and print shape of array
    shape = img_np_arr.shape
    print(shape)

    #Create a video writer
    screen_cap_writer = cv2.VideoWriter('screen_recorded.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 50, (shape[1], shape[0]))

    #To View the screen recording in a separate window (OPTIONAL)
    #This is optional. Use the aspect ratio scaling if you wish to view the screen recording simultaneously
    #Low scale_by_percent implies smaller window
    scale_by_percent = 50
    width = int(shape[1] * scale_by_percent / 100)
    height = int(shape[0] * scale_by_percent / 100)
    new_dim = (width, height)

    #Record the screen
    #Condition to keep recording as a video
    while True:
       #Capture screen
       image = ImageGrab.grab()
       #Convert to array
       img_np_arr = np.array(image)
       #OpenCV follows BGR and not RGB, hence we convert
       final_img = cv2.cvtColor(img_np_arr, cv2.COLOR_RGB2BGR)
       #Write to video
       screen_cap_writer.write(final_img)
       #OPTIONAL: To view your screen recording in a separate window, resize and use imshow()

       '''
           If you choose to view the screen recording simultaneously,
           It will be displayed and also recorded in your video.
       '''
       image = cv2.resize(final_img, (new_dim))
       cv2.imshow("image", image)

       #Stop and exit screen recording if user presses 'e' (You can put any letter)
       if cv2.waitKey(1) == ord('e'):
           break
      
    #Release the created the objects
    screen_cap_writer.release()
    cv2.destroyAllWindows()
    #Define the user interface for Screen Recorder using Python

class Webbrowser_PythonEdition(QMainWindow):

	# constructor
	def __init__(self, *args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)


		# creating a QWebEngineView
		self.browser = QWebEngineView()

		# setting default browser url as google
		self.browser.setUrl(QUrl("http://www.bing.com"))

		# adding action when url get changed
		self.browser.urlChanged.connect(self.update_urlbar)

		# adding action when loading is finished
		self.browser.loadFinished.connect(self.update_title)

		# set this browser as central widget or main window
		self.setCentralWidget(self.browser)

		# creating a status bar object
		self.status = QStatusBar()

		# adding status bar to the main window
		self.setStatusBar(self.status)

		# creating QToolBar for navigation
		navtb = QToolBar("Navigation")

		# adding this tool bar tot he main window
		self.addToolBar(navtb)

		# adding actions to the tool bar
		# creating a action for back
		back_btn = QAction("Back", self)

		# setting status tip
		back_btn.setStatusTip("Back to previous page")

		# adding action to the back button
		# making browser go back
		back_btn.triggered.connect(self.browser.back)

		# adding this action to tool bar
		navtb.addAction(back_btn)

		# similarly for forward action
		next_btn = QAction("Forward", self)
		next_btn.setStatusTip("Forward to next page")

		# adding action to the next button
		# making browser go forward
		next_btn.triggered.connect(self.browser.forward)
		navtb.addAction(next_btn)

		# similarly for reload action
		reload_btn = QAction("Reload", self)
		reload_btn.setStatusTip("Reload page")

		# adding action to the reload button
		# making browser to reload
		reload_btn.triggered.connect(self.browser.reload)
		navtb.addAction(reload_btn)

		# similarly for home action
		home_btn = QAction("Home", self)
		home_btn.setStatusTip("Go home")
		home_btn.triggered.connect(self.navigate_home)
		navtb.addAction(home_btn)

		# adding a separator in the tool bar
		navtb.addSeparator()

		# creating a line edit for the url
		self.urlbar = QLineEdit()

		# adding action when return key is pressed
		self.urlbar.returnPressed.connect(self.navigate_to_url)

		# adding this to the tool bar
		navtb.addWidget(self.urlbar)

		# adding stop action to the tool bar
		stop_btn = QAction("Stop", self)
		stop_btn.setStatusTip("Stop loading current page")

		# adding action to the stop button
		# making browser to stop
		stop_btn.triggered.connect(self.browser.stop)
		navtb.addAction(stop_btn)

		# showing all the components
		self.show()


	# method for updating the title of the window
	def update_title(self):
		title = self.browser.page().title()
		self.setWindowTitle("% s -  Browser" % title)


	# method called by the home action
	def navigate_home(self):

		# open the google
		self.browser.setUrl(QUrl("http://www.bing.com"))

	# method called by the line edit when return key is pressed
	def navigate_to_url(self):

		# getting url and converting it to QUrl object
		q = QUrl(self.urlbar.text())

		# if url is scheme is blank
		if q.scheme() == "":
			# set url scheme to html
			q.setScheme("http")

		# set the url to the browser
		self.browser.setUrl(q)

	# method for updating url
	# this method is called by the QWebEngineView object
	def update_urlbar(self, q):

		# setting text to the url bar
		self.urlbar.setText(q.toString())

		# setting cursor position of the url bar
		self.urlbar.setCursorPosition(0)
class basicNeuralNetwork1:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
def websiteconn():
    windowwebconn=Tk()
    # Set the size of the tkinter window
    windowwebconn.geometry("700x350")
    windowwebconn.title("Website Connectivity Checker")#give title to the window
    url=tk.StringVar()# url is of string type
    def check():
        try:
            web= (url.get())
            status_code = urllib.request.urlopen(web).getcode()
        except urllib.error.URLError:
            Label(windowwebconn, text="Website Couldn't Be Reached", font=('Calibri 15')).place(x=260,y=200)
        try:
            website_is_up = status_code == 200
        except UnboundLocalError as e:
            _a=str(e)
            if _a=="UnboundLocalError: local variable 'status_code' referenced before assignment":
                return
        if website_is_up==TRUE:
            Label(windowwebconn, text="Website Available", font=('Calibri 15')).place(x=260,y=200)
        else:
            Label(windowwebconn, text="Website Not Available", font=('Calibri 15')).place(x=260,y=200)
    head=Label(windowwebconn, text="Website Connectivity Checker", font=('Calibri 15'))# a label
    head.pack(pady=20)
    Entry(windowwebconn, textvariable=url).place(x=200,y=80,height=30,width=280)# enter a website url
    #create a button
    Button(windowwebconn, text="Check",command=check).place(x=320,y=160)
    windowwebconn.mainloop()#main command
def vid2audio():
    windowva=Tk()
    # Set the size of the tkinter window
    windowva.geometry("700x350")
    windowva.title("VIDEO TO AUDIO CONVERTER")#give title to the window
    Label(windowva, text="VIDEO TO AUDIO CONVERTER",bg='orange', font=('Calibri 15')).pack()# a label
    Label(windowva, text="Choose a File ").pack()
    pathlab = Label(windowva)
    pathlab.pack()
    def browse():#browsing function
        global video#global variable
        video = askopenfilename()
        video = moviepy.editor.VideoFileClip(video)
        pathlab.config(text=video)#configure method
    def save():
        audio = video.audio#convert to audio
        audio.write_audiofile("sample.wav")#save as audio
        Label(windowva, text="Video Converted into Audio and Saved Successfully",bg='blue', font=('Calibri 15')).pack()# a label
    #creating buttons
    Button(windowva,text='browse',command=browse).pack()
    Button(windowva,text='SAVE',command=save).pack()
    windowva.mainloop()


def qr_codegen():
    windowqrcode = Tk()  
    windowqrcode.geometry('300x350')
    windowqrcode.title('QR code generator')

    Label(windowqrcode,text='Lets Create QR Code',font='arial 15').pack()
    # String which represents the QR code
    s = tk.StringVar()
      
    # Generate QR code
    def create_qrcode():
        s1=s.get()
        qr = pyqrcode.create(s1)
        qr.png('myqr.png', scale = 6)
        Label(windowqrcode,text='QR Code is created and saved successfully').pack()

    Entry(windowqrcode,textvariable=s,font='arial 15').pack()
    Button(windowqrcode,text='create',bg='pink',command=create_qrcode).pack()
    windowqrcode.mainloop()
class MainWindow(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setGeometry(100, 100, 300, 50)
        self.setWindowTitle('Click me to start!')

        layout = QVBoxLayout()
        self.setLayout(layout)

        hbox = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        hbox.addWidget(self.progress_bar)

        layout.addLayout(hbox)

        hbox = QHBoxLayout()
        self.btn_progress = QPushButton('Start AI', clicked=self.progress)

        # align buttons center
        hbox.addStretch()
        hbox.addWidget(self.btn_progress)
        hbox.addStretch()
        
        layout.addLayout(hbox)
        self.current_value = 0
        self.show()



    def progress(self):
        if self.current_value  <= self.progress_bar.maximum():
            for i in range(20):
                self.current_value += 5
                self.progress_bar.setValue(self.current_value)
                time.sleep(0.05)
            return


def m():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()
m()
        
_hello_words=[
    "Hello,how can i assist you today?",
    "Good morning/afternoon/evening",
    "Hello:D"
    ]
_bye_words=[
    "Goodbye! If you have any non-political or non-sexual questions in the future, feel free to ask.",
    "Goodbye! Have a great day!",
    "Ok, if you have any other questions or need help, please feel free to let me know, I will be happy to serve you.Have a nice day!"
    ]
_responding_user_inputsqrcode=[
    "create qr code python",
    "create qr codes python",
    "Create QR codes python",
    "Create QR code python",
    "Create qr codes python",
    "create QR codes python"
    ]
_responding_user_inputs_vidaudconv=[
    "video to audio converter python"
    ]
_responding_user_inputs_checkwebsiteconnectivity=[
    "Python Site Connectivity Checker",
    "python check website connectivivity"
    ]
_responding_user_inputs_basicneauralnetwork=[
    "NeuralNetwork",
    "Neural Network"
    "neuralnetwork",
    "neural network",
    "neural Network"
    ]
_responding_user_inputs_webbrowserinpython=[
    "web browser in python",
    ]
_responding_user_inputs_screenrecorderpython=[
    "screen recorder python"
    ]


goga1=0
print("GPT:"+str(_hello_words[random.randint(0,2)]))
while True:
    try:
        user_input=str(input("user>>>"))
    except KeyboardInterrupt:
        print("GPT:"+str(_bye_words[random.randint(0,2)]))
        sys.exit(0)
    if user_input in _responding_user_inputsqrcode:
        print("GPT:running qr code creator...")
        qr_codegen()
        print("GPT:saved a myqr.png in your local folder!")
    elif user_input in _responding_user_inputs_vidaudconv:
        print("GPT:running video to audio converter")
        vid2audio()
    elif user_input in _responding_user_inputs_checkwebsiteconnectivity:
        print("GPT:running network connectivity checker..")
        websiteconn()
    elif user_input in _responding_user_inputs_basicneauralnetwork:
        print("A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the",end="")
        time.sleep(0.1)
        print("human brain. It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a",end="")
        time.sleep(0.1)
        print("layered structure that resembles the human brain. It creates an adaptive system that computers use to learn from their mistakes",end="")
        time.sleep(0.1)
        print("and improve continuously. Thus, artificial neural networks attempt to solve complicated problems, like summarizing documents",end="")
        time.sleep(0.1)
        print("or recognizing faces, with greater accuracy.")
        time.sleep(0.1)
        print("-------would you like some examples on neual network?(yes or no)")
        _user_input=str(input("?user>>>"))
        if _user_input=="yes":
            print("1)basic machine learning(learning rate 0.1)")
            print("select.")
            _user_input=str(input("?user>>>"))
            if _user_input=="1":
                input_vectors = np.array(
                    
                    [
                        [3, 1.5],
                        [2,1],
                        [4,1.5],
                        [3,4],
                        [3.5,0.5],
                        [2,0.5],
                        [5.5,1],
                        [1,1],
                    ]
                )
                
                targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
                learning_rate = 0.1
                neural_network = basicNeuralNetwork1(learning_rate)
                training_error = neural_network.train(input_vectors, targets, 10000)
                plt.plot(training_error)
                plt.xlabel("Iterations")
                plt.ylabel("Error for all training instances")
                print("done saving..")
                plt.savefig("cumulative_error.png")
            else:
                pass
        else:
            pass
    if user_input in _responding_user_inputs_webbrowserinpython:
        print("loading webbrowser(python edition)")
        # creating a pyQt5 application
        app = QApplication(sys.argv)

        # setting name to the application
        app.setApplicationName("Browser")

        # creating a main window object
        window = Webbrowser_PythonEdition()

        # loop
        app.exec_()
    if user_input in _responding_user_inputs_screenrecorderpython:
        screenrecordergui()
    else:
        print("GPT:i don't know what you're talking about please say again!")

        
        
    
    
