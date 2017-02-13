"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Program to down load data from a url and copy in pwd

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"""


import urllib

# Take note of the pwd in the terminal as that is where the file will be copied to

# put target url here
url = "https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py"

# what do you want to call the file?
filename = "input_data.py"

#call the urlretrieve function and pass in arguments
tmp_file = urllib.URLopener()
tmp_file.retrieve(url, filename)
