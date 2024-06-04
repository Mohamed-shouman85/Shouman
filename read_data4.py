import os  # Import OS library for operating system interactions (not used in the provided script)
import pandas as pd  # Import Pandas library for data manipulation and analysis
import numpy as np  # Import NumPy library for numerical operations
from pathlib import Path  # Import Path class from pathlib module for filesystem paths
import re  # Import regular expression module for pattern matching
import pickle
import time

class Dataset:
    def __init__(self):
        # Initialize empty lists to store various types of data
        self.Irradiance = []  # List to store irradiance values
        self.Temperature = []  # List to store temperature values
        self.MPP = []  # List to store Maximum Power Point (MPP) values
        self.Power = None  # List to store Power Values
        self.Voltage = None  # List to store voltage readings
        self.Current = None  # List to store current readings
    
    def updatedata(self, filepath):
        # Method to update dataset with new data from a specified file path
        Temperature, Irradiance, I, V = self.getFeatures(filepath)  # Extract features and IV data from file
        self.Irradiance.append(Irradiance)  # Append new irradiance value
        self.Temperature.append(Temperature)  # Append new temperature value
        MPP = int(round(np.max(I * V)))  # Calculate and round the max product of I and V to get MPP
        Power = I*V
        self.MPP.append(MPP)  # Append calculated MPP
        num_values = 10000  # Define the number of values to be considered for I and V
        if len(V) < num_values:  # Check if there are enough voltage readings
            raise ValueError("Not enough unique elements to fill the request.")
        Indices = self.select_indices_with_product_max_numpy(V, I, num_values)  # Select indices based on max product criteria
        if self.Voltage is None:
            self.Voltage = (np.array(V[Indices].reshape(1,-1)))
        else:
            self.Voltage = np.vstack((self.Voltage, V[Indices].reshape(1,-1)))
        if self.Current is None:
            self.Current = (np.array(I[Indices].reshape(1,-1)))
        else:
            self.Current = np.vstack((self.Current, I[Indices].reshape(1,-1)))
        if self.Power is None:
            self.Power = (np.array(Power[Indices].reshape(1,-1)))
        else:
            self.Power = np.vstack((self.Power, Power[Indices].reshape(1,-1)))
        #self.Voltage.append((np.array(V[Indices]).T))  # Append selected voltage values based on Indices
        #self.Current.append(I[Indices])  # Append selected current values based on Indices
        #self.Power.append(Power[Indices])  # Append selected power values based on Indices
        #print(type(V[Indices]))  # Print the shape of the selected voltage array
        #time.sleep(100)

    def select_indices_with_product_max_numpy(self, data1, data2, num_values):
        # Select indices for min, max, and max product from two data arrays
        if data1.shape != data2.shape:  # Ensure both arrays are of the same size
            raise ValueError("Both data arrays must have the same length.")
        if num_values < 3:  # Ensure at least three values are requested to include min, max, and max product
            raise ValueError("num_values must be at least 3 to include min, max, and max product indices.")
        if len(data1) < num_values:  # Ensure there are enough elements in the data arrays
            raise ValueError("num_values must be less than or equal to the length of the data.")
        product = data1 * data2  # Calculate the product of the two arrays
        max_product_idx = np.argmax(product)  # Find the index of the maximum product
        indices = np.arange(len(data1))  # Generate an array of indices for data1
        filtered_indices = np.delete(indices, [max_product_idx])  # Remove the index of the max product from consideration
        data1f = data1[filtered_indices]  # Create a filtered array of data1 excluding the max product index
        min_idx = np.argmin(data1f)  # Find the index of the minimum value in the filtered data1 array
        max_idx = np.argmax(data1f)  # Find the index of the maximum value in the filtered data1 array
        indices = np.arange(len(data1))  # Re-generate the array of indices for data1
        filtered_indices = np.delete(indices, [min_idx, max_idx])  # Remove the indices of min and max values
        random_indices = np.random.choice(filtered_indices, num_values - 3, replace=False)  # Randomly select the remaining indices
        result_indices = np.array([min_idx, max_idx, max_product_idx] + random_indices.tolist()) 
        sorted_indices = np.sort(result_indices)  # Sort by values of data1
        return sorted_indices  # Return the shuffled array of selected indices

    def GetIV(self, filepath):
        # Extract Current (I) and Voltage (V) values from an Excel file based on a provided file path
        V = np.array(pd.read_excel(filepath, sheet_name='DataMatrix', usecols='A', skiprows=[0]))  # Read and convert Voltage data from Excel
        I = np.array(pd.read_excel(filepath, sheet_name='DataMatrix', usecols='B', skiprows=[0]))  # Read and convert Current data from Excel
        return V, I  # Return the arrays of Voltage and Current

    def generatefilepaths(self, folderpath):
        # Generate file paths for all Excel files within a given folder
        directory = Path(folderpath)  # Create a Path object for the specified folder
        paths = list(directory.rglob('*.xlsx'))  # List all Excel files recursively within the folder
        paths_as_strings = [str(path) for path in paths]  # Convert Path objects to strings
        return paths_as_strings  # Return the list of file paths as strings

    def getFeatures(self, filepath):
        # Extract features like temperature and irradiance from a file path using regular expressions
        patterns = ['(\d+)\s*deg', '(\d+)\s*irr']  # Define patterns to find temperature and irradiance in the file path
        Temperature = None  # Initialize Temperature as None
        Irradiance = None  # Initialize Irradiance as None
        for index, pattern in enumerate(patterns):  # Iterate over each pattern
            match = re.search(pattern, filepath)  # Search for the pattern in the filepath
            if match:  # If a match is found
                if index == 0:  # If the match is for temperature
                    Temperature = int(match.group(1))  # Convert the matched string to an integer
                elif index == 1:  # If the match is for irradiance
                    Irradiance = int(match.group(1))  # Convert the matched string to an integer
            else:  # If no match is found
                if index == 0:  # If temperature was expected
                    Temperature = int(input("No match found for Temperature, Select Value: "))  # Prompt for manual input
                elif index == 1:  # If irradiance was expected
                    Irradiance = int(input("No match found for Irradiance, Select Value: "))  # Prompt for manual input
        I, V = self.GetIV(filepath)  # Retrieve Current and Voltage data
        return Temperature, Irradiance, I, V  # Return the extracted features and data

    def sendfolderpath(self, folderpath):
        # Process all Excel files within a specified folder to update the dataset
        path_files = self.generatefilepaths(folderpath)  # Generate file paths for all Excel files
        for path in path_files:  # Loop through each file path
            print(path)  # Print the current file path
            self.updatedata(path)  # Update the dataset with data from the current file

    def sendfilepath(self, filepath):
        # Update the dataset with data from a single file path
        self.updatedata(filepath)  # Call updatedata method with the specified file path

    def save(self, datasetname):
        # Save the dataset with data into specific file path
        with open(datasetname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self,datasetname):
        # load the dataset with data into specific file path
        with open(datasetname, 'rb') as input:
            loaded_object = pickle.load(input)
            self.Irradiance=loaded_object.Irradiance
            self.Temperature=loaded_object.Temperature
            self.MPP=loaded_object.MPP
            self.Voltage=loaded_object.Voltage
            self.Current=loaded_object.Current
            self.Power=loaded_object.Power

'''            
if __name__ == '__main__':
    Data1 = Dataset()  # Create an instance of Dataset
    print('Irradiance before load = ',Data1.Irradiance)
    Data1.load('my_class.pkl')
    print('Irradiance after load = ',Data1.Irradiance)
    
    FolderPath = 'sim-data'  # Define the folder path containing data files
    Data1.sendfolderpath(FolderPath)  # Process all files in the specified folder
    print('Irradiance Values:', Data1.Irradiance)  # Print all collected irradiance values
    print('Temperature Values:', Data1.Temperature)  # Print all collected temperature values
    print('MPP Values:', Data1.MPP)  # Print all collected MPP values
    Data1.save()
'''

    # Uncomment the next line to print Voltage values stored in the dataset
    # print('Voltage Values:', np.array(Data1.Voltage))
