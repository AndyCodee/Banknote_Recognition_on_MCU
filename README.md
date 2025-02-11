# Using NuMaker-M032KG, combine a neural network to Recognize the texture of banknote
- In this project, we aim to utilize three photoresistors to measure the light intensity corresponding to RGB values. This is achieved by covering each photoresistor with a red, green, or blue translucent filter, allowing only the respective color’s light to pass through. By doing so, we can capture the intensity of each RGB component and analyze variations in light intensity to identify color pattern changes on banknotes.
- The overall methodology involves first collecting data, then using this time-series data to train a simple neural network. The trained model outputs a weight matrix, which is subsequently used in the final predictive model for real-time banknote color pattern recognition.
- gif


## Content
* [Device and IDE](#device-and-ide)
* [Circuit Diagram](#circuit-diagram)
* [Method](#method)
* [Machine Learning](#machine-learning)
* [Flow Chart](#flow-chart)
* [DEMO Video (YouTube)](#demo-video-youtube)

## Device and IDE
### [NuvoTon NuMaker-M032KG](https://direct.nuvoton.com/tw/numaker-m032kg)
![numaker-m032kg](https://github.com/user-attachments/assets/53c2646d-d427-4818-993f-16b76a3c903f)

### [keil uVision 5](https://www.keil.com/download/list/uvision.htm)
<img src="https://github.com/user-attachments/assets/f8142b6a-61fc-459c-83a1-2361ee4d3eb9" width="200">


## Circuit Diagram
![image](https://github.com/user-attachments/assets/65bda780-0380-43b8-a790-017e51f4f0d4)

## Method
### Operating Principle
`Color Sensor`:
- When an object is illuminated with white light, it reflects 
the wavelengths of light that correspond to its inherent 
color.
- Reflected light can be used to determine an object’s color.
- A photoresistor’s resistance varies with light intensity
- Cellophane can assist in filtering light
- Combining these helps detect the intensity of specific 
colored light
- ![image](https://github.com/user-attachments/assets/d53c70fb-e76b-4c94-8a99-273e58268411)

### Machine Learning
- Use MLP (Multilayer Perceptron) as the model architecture.
- ![image](https://github.com/user-attachments/assets/1208b72e-beb0-4a88-8600-db722dd545b8)

### Flow Chart
![image](https://github.com/user-attachments/assets/26ee4f93-4d54-4b3a-afb6-f2aa02374a62)


## DEMO Video (YouTube) 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/qb9uLU0ng0Y/0.jpg)](https://www.youtube.com/watch?v=qb9uLU0ng0Y)

