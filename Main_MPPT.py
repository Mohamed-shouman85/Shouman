# Main.py
# This is the main file that imports functions and configures settings from other files.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Imports
import datetime
import numpy as np
#import pandas as pd
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from numpy.polynomial.polynomial import Polynomial
from tensorflow.keras.models import Model
import tkinter as tk
from tkinter import filedialog, simpledialog
import tf2onnx



from read_data4 import Dataset  # Getting Dataset class read_data4.py is in your Python path
from model_training_keras_MPPT import Train_Dataset, Test_req  # Training or Retraining the Dataset to generate class read_data4.py is in your Python path

Data=Dataset()

# Get the current date and time
current_time = datetime.datetime.now()

# Format the date and time to exclude the year (MM-DD_HH-MM-SS)
formatted_time = current_time.strftime("%m-%d-%H-%M")

print(formatted_time)
#from read_data4 import Dataset  # Getting Dataset class read_data4.py is in your Python path
#from model_training_keras2 import Dataset_Train  # Training or Retraining the Dataset to generate class read_data4.py is in your Python path

root = tk.Tk()
root.title("MPPT GUI")
root.geometry("800x600")  # Set initial size of the window


def load_old_dataset():
    global Data
    print('Before Inside1 = ', Data.Irradiance)
    filetype = [('Pkl files', '*.pkl')]
    filepath = filedialog.askopenfilename(filetypes=filetype)
    if filepath:
        print(filepath)
        Data.load(str(filepath))
    #print('After Inside1 = ', Data.Irradiance)

def update_with_file():
    global Data
    #print('Before Inside2 = ', Data.Irradiance)
    filetype = [('Excel files', '*.xlsx')]
    filepath = filedialog.askopenfilename(filetypes=filetype)
    if filepath:
        print(filepath)
        Data.sendfilepath(str(filepath))
    #print('After Inside2 = ', Data.Irradiance)


def update_with_folder():
    global Data
    #print('Before Inside3 = ', Data.Irradiance)
    folderpath = filedialog.askdirectory()
    if folderpath:
        print(folderpath)
        Data.sendfolderpath(str(folderpath))
    #print('After Inside3 = ', Data.Irradiance)
    #print('Voltage After Inside3 = ', Data.Voltage.shape)

def save_new_dataset():
    global Data
    #print('Before Inside4 = ', Data.Irradiance)
    # Create the filename with the formatted date and time
    filename = f"data_{formatted_time}.pkl"
    Data.save(str(filename))
    #print(f"save_dataset")

def train_dataset():
    global Data
    #print('Before Inside5 = ', Data.Irradiance)
    filename = f"pre_train_MPPT_{formatted_time}.h5"
    print(filename)
    model = Train_Dataset(Data, filename)
    #print(f"Training based on Polynomial Degree = {Eqn_degree}")

def load_trained_dataset():
    filetype = [('H5 files', '*.h5')]
    filepath = filedialog.askopenfilename(filetypes=filetype)
    if filepath:
        print(filepath)
    model = load_model(filepath)

def test_trained_dataset():
    filetype = [('H5 files', '*.h5')]
    filepath = filedialog.askopenfilename(filetypes=filetype)
    if filepath:
        print(filepath)
    model = load_model(filepath)
    num1 = int(input1.get())
    num2 = int(input2.get())
    Xtest= np.array([num1,num2]).reshape((1, 2))
    print("Shape of generated Tested X:" , Xtest.shape)
    MPP = Test_req(model, Xtest)
    print('Predicted in Main = ', MPP)
    # Display the result in the text editor
    output.insert(tk.END, str(MPP[0]))

    # Transform tensorflow model to ONNX for Matlab
    keras_model = tf.keras.models.load_model(filepath)

    # Convert to TensorFlow format
    spec = (tf.TensorSpec((None, 2), tf.float32, name="input"),)  # Adjust input_size as needed
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=None)
    print("Checked" , filepath)

    # Save the ONNX model
    with open(filepath.replace(".h5", ".onnx"), "wb") as f:
        f.write(model_proto.SerializeToString())
    print("all done")

def close_gui():
    print('Closing GUI')
    root.destroy()
    
# Main function definition


print('Before Outside = ', Data.Irradiance)
# Setup the main window
Load = tk.Button(root, text="Load Old Dataset", command=load_old_dataset)
Load.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

Update_file = tk.Button(root, text="Update with Excel File", command=update_with_file)
Update_file.grid(row=1, column=1, padx=20, pady=20, sticky="ew")

Update_folder = tk.Button(root, text="Update with Folder", command=update_with_folder)
Update_folder.grid(row=1, column=3, padx=20, pady=20, sticky="ew")

Save_dataset = tk.Button(root, text="Save New Dataset", command=save_new_dataset)
Save_dataset.grid(row=2, column=2, padx=20, pady=20, sticky="ew")

Train_dataset = tk.Button(root, text="Train Dataset", command=train_dataset)
Train_dataset.grid(row=3, column=1, columnspan=1, padx=20, pady=20, sticky="ew")

Load_trained_dataset = tk.Button(root, text="Load Trained Dataset", command=load_trained_dataset)
Load_trained_dataset.grid(row=3, column=3, columnspan=1, padx=20, pady=20, sticky="ew")

Input_label1 = tk.Label(root, text="Irradiance")
Input_label1.grid(row=5, column=0, columnspan=1, padx=20, pady=20)
    
input1 = tk.Entry(root, width=10)
input1.grid(row=5, column=1, columnspan=1, padx=20, pady=20)

input_label2 = tk.Label(root, text="Temperature")
input_label2.grid(row=5, column=2, columnspan=1, padx=20, pady=20)
    
input2 = tk.Entry(root, width=10)
input2.grid(row=5, column=3, columnspan=1, padx=20, pady=20)
    
Test_data = tk.Button(root, text="Test Trained Dataset", command=test_trained_dataset)
Test_data.grid(row=6, column=1, columnspan=1, padx=20, pady=20, sticky="ew")

result_label = tk.Label(root, text="MPP Result")
result_label.grid(row=6, column=2, columnspan=1, padx=20, pady=20)
    
output = tk.Entry(root, width=10)
output.grid(row=6, column=3, columnspan=1, padx=20, pady=20)

Close_gui = tk.Button(root, text="Close GUI", command=close_gui)
Close_gui.grid(row=7, column=2, columnspan=1, padx=20, pady=20, sticky="ew")
    
root.mainloop()


