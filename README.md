# motion-heatmap-opencv
This project detects human figures and generates a heatmap based on their pathing. 
It is a proof of concept and limited in current application due to a lack of data storage implementation. 

It is based off the code from [Medium](https://medium.com/p/fd806e8a2340) article which tracks all motion, which is itself an adaptation of the [Intel Motion Heatmap](https://github.com/intel-iot-devkit/python-cv-samples/tree/master/examples/motion-heatmap)

![example gif](./example/outputMALL_square2.gif)

# Run
Clone this repository, `cd` into the directory and run `python motion_heatmap3.py `. 
If you want to use another video, change the desired video's name to `input.avi` or alter `motion_heatmap.py` in the `main()` function.
Individual frames will be created within the `frames` folder that is created before being compiled into the `output` folder.
`example` folder contains an example of the final output. 

# Requirements
To run this script you will need python 3.6+ installed along with OpenCV 3.3.0+ and numpy.
Make also sure to have installed the MOG background subtractor by running:

`pip install opencv-contrib-python`

`pip install progress` for creating a basic visual progress bar in the terminal. 

# Enjoy!
