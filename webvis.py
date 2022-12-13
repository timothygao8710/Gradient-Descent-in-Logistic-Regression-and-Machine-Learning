#pywebio for display on website
import pywebio
from pywebio.input import input, FLOAT, input_group, NUMBER
from pywebio.output import put_text, put_html, put_markdown, put_button, put_link

#plotly and matplotlib to assist with 3d graphing
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt

# numpy for fast math operations
import numpy as np 
# numpy to support more user-inputted functions
from numpy import sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, log, gcd, sinc

# basic classes for utility
from math import ceil
from random import randrange

# flask to deploy website to heroku
from flask import Flask, send_from_directory

# argparse to help set up port
import argparse

# select a website theme (make it prettier)
pywebio.config(theme="minty")

#initialize global variables
x, x_min, x_max, x_samples, x_vals = 0,0,0,0,0 
y, y_min, y_max, y_samples, y_vals = 0,0,0,0,0
equation, z = 0, 0
#Used for calculating learning rate
#Inversely proportional to number of steps
alpha = 0.015

#calculate initial learning rate, modifying user's input based on x_samples and y_samples, as well as an experimentally determined alpha
#we need to do this because LR is more insignificant as the number of samples we take increases
def calcLR():
    return 0.05 * alpha * (x_samples + y_samples)

#Function for changing learning rate: see "Main Paper" for more details
#a is some coefficient, can be tuned
#Other options include LR - a, LR/(1 + a * LR), LR ^ a
def updateLR(LR, a = 0.3):
    return LR * np.exp(-a)
    #LR over iterations graph becomes: e ^ (- x * a)

# Make sure user's entered domain is valid
def check_range(val):
    if val > 50:
        return 'Please choose a value less than 50 for detailed graph'
    elif val < -50:
        return 'Please choose a value larger than -50 for detailed graph'

# Make sure user's entered learning rate is valid
def check_LR(val):
    if val < 0 or val > 1:
        return "Please enter a percentage between 0 and 1"
    
# make sure user's entered equation is valid
def check_eq(eq_str):
    eq_str = eq_str.replace('^', '**')
    try:
        for x in range(-50, 50, 10):
            for y in range(-50, 50, 10):
                eval(eq_str)
    except Exception as e:
        return 'Please enter a valid expression: ' + str(e)

# drawing vectors on the 3D graph to demonstrate gradient descent
def draw_vectors(step_history):
    # array to store all objects to plot
    plotted = []
    
    # relevant parameters for the cone and line making up each vector
    x_locs, y_locs, z_locs = [], [], []
    x_delta, y_delta, z_delta = [], [], []
    
    # kinda useless now but uncommenting later code enables plotting of a dot at final point
    result_dot = 0
    
    # we iterate through the step history we took during GD
    for i in range(len(step_history)):
        #this section computes relevant differences between previous point and current point, to find the vector that enables us to travel between the two
        xi, yi = step_history[i][0], step_history[i][1]
        xn, yn = 0, 0
        
        if i != len(step_history)-1:
            xn, yn = step_history[i+1][0], step_history[i+1][1]
        else:
            xn, yn = xi, yi
        
        zi, zn = z[xi][yi], z[xn][yn]
        xi, yi, xn, yn = x_vals[xi], y_vals[yi], x_vals[xn], y_vals[yn]
        
        dx, dy, dz = xn - xi, yn - yi, zn - zi
        
        # actually append out computed results to the arrays, later using them to graph
        x_locs.append(xi)
        y_locs.append(yi)
        z_locs.append(zi)
        
        x_delta.append(dx)
        y_delta.append(dy)
        z_delta.append(dz)
        
        # kinda useless now but uncommenting later code enables plotting of a dot at final point
        if i == len(step_history)-1:
            result_dot = (xi, yi, zi)
    
    # we first make the cone objects for the head of the vectors, scaled to the size of each vector's magnitude
    cones = go.Cone(
        x=x_locs,
        y=y_locs,
        z=z_locs,
        u=x_delta,
        v=y_delta,
        w=z_delta,
        sizemode="scaled",
        sizeref= 0.5,
        anchor='tail'
        , showscale=False
    )
    
    # we then make the line objects for the lines of the vectors, scaled to the size of each vector's magnitude
    lines = go.Scatter3d(
        x=x_locs,
        y=y_locs,
        z=z_locs,
        mode='lines',
        line=dict(
            color='red',
            width=2
        )
    )

#     dot = go.Scatter3d(
#         x=[result_dot[0]],
#         y=[result_dot[1]],
#         z=[result_dot[2]],
#         mode='markers'
#     )
    
    # put both objects inside the array of objects to be plotted
    plotted.append(cones)
    plotted.append(lines)
#     plotted.append(dot)

    return plotted

#the main gradient descent algorithm
def gradient_descent(x_init, y_init, max_steps = 500):
    #calculate the learning rate
    LR = ceil(calcLR())
    # set (xi, yi), the current point, as (x_init, y_init) the initial points
    xi, yi = x_init, y_init
    #res stands for "result", stores the points we iterated through
    res = []
    
    #iterate through max_steps number of steps
    for i in range(max_steps):
        res.append((xi, yi))
        
        #rather than taking the derivative directly, in this visualization we scan a LRxLR square to find the least value
        #then we actually walk in that value
        #each cell in the LRxLR square is a sample that we took based on specific x/y values
        #the idea is the same, except we can't directly use derivative because our samples are discrete
        best_deriv = (0, 0, 0)
        
        #iterate through all cells in the square
        for j in range(-LR, LR+1):
            for k in range(-LR, LR+1):
                #find the indices of the cell
                nx, ny = xi + j, yi + k
                #check if they are within logical constraints
                if nx < 0 or ny < 0 or nx >= x_samples or ny >= y_samples:
                    continue
                
                #update based on wether or not its better (lower minimum) than best deriv
                cur_deriv = (z[nx][ny] - z[xi][yi], nx, ny)
                best_deriv = min(best_deriv, cur_deriv)
        
        #shtop if we don't move
        if best_deriv[0] == 0:
            break
        
        #get the new (xi, yi), we have now taken a step
        xi = best_deriv[1]
        yi = best_deriv[2]

        #we actually update LR to be lower, based on a simulated-annealing-inspired-equation
        #See "Main Paper" notebook for more details
        LR = ceil(updateLR(LR))

    #return a tuple of the minimum value as well as the points we stepped through
    return (z[xi][yi], res)

#getting all the objects we need to plot ready
def getFig():
    
    #this section gets the gradient descent points, by calling it at many random points
    best_res = (np.max(z), [])
    for i in range(250):
        #start at a random point
        xi, yi = randrange(x_samples), randrange(y_samples)
        #descent from this point
        cur_res = gradient_descent(xi, yi)
        #use this one if it's the global min
        if cur_res[0] < best_res[0] or (len(cur_res[1]) > len(best_res[1])):
            best_res = cur_res
    
    # this section gets all the vectors based on the gradient descent points
    plotted = draw_vectors(best_res[1])

    # this section gets the 3D graph object
    plotted.append(go.Surface(x=x, y=y, z=z, colorscale = 'viridis', opacity=0.6, showscale=False))

    # this section makes the figure itself, using all the objects we acculumated
    fig = go.Figure(data=plotted)
    # this stands for "minimum value", the best (xi, yi) we found
    mv = best_res[1][len(best_res[1])-1]
    
    # camera = dict(
    #     eye=dict(x=x_vals[mv[0]] * 0.8, y=y_vals[mv[1]] * 0.8, z = np.max(z) + 0.05 * (np.max(z) - np.min(z)))
    # )

    # add title to the figure
    fig.update_layout(title= "f(x,y) = " + equation.replace('**', '^'))

    # return the figure as well as texts to be displayed of our minimum value point and the number of steps it took to converge
    return (
        fig,
        "Minimum value at: (" + str(round(x_vals[mv[0]], 4)) + ", " + str(round(y_vals[mv[1]], 4)) + ", " + str(round(best_res[0], 4)) + ")",
        "Converged in " + str(len(best_res[1])-1) + " steps with initial learning rate of " + str(alpha)
    )

def web():
    #putting a bunch of text and links
    put_text("Gradient Descent Visualization").style('font-size: 40px')
    put_text("by Timothy Gao").style('font-size: 15px')
    
    put_text("This is a visualization of the gradient descent algorithm on an interactive 3D graph. Users can customize the graph by entering any equation for the function f(x, y). Please use the '*' symbol for multiplication. Also, feel free to experiment with functions like 'tanh', 'abs', and 'log', which are all supported. Users can also customize the maximum and minimum values of the function as well as the learning rate. After graphing, a visual of both the graph and each iteration of gradient descent will be displayed, where vectors are scaled to the magnitude of each step and will be shown on a 3D graph. The user can then use their cursor to rotate the graph, zoom in and out (by scrolling), pan the graph, take a screenshot, and play around with different camera angles. The minimum value found by gradient descent and total number of iterations to reach the minimum will also be displayed at the bottom.").style('font-size: 15px')
    put_link(name="Please visit the notebook for more details", url="https://github.com/timothygao8710/Applications-of-Gradient-Descent-in-Machine-Learning/blob/main/Logistic%20Regression.ipynb", new_window=True)
    
    #putting this here to access global vars
    global x_min, x_max, x_samples, x_vals, y_min, y_max, y_samples, y_vals, x, y, z, alpha, equation
    
    #this allows us to get user input and check if they're valid
    info = input_group("Customize Graph",[
        input('f(x, y) = ', name='equation', value = "sin(x) ^ 10 + cos(10 + y * x) * cos(x)", validate = check_eq),
        input('Minimum value: ', name='min', type=FLOAT, value = "-3", validate = check_range),
        input('Maximum value: ', name='max', type=FLOAT, value = "3", validate = check_range),
        input('Learning rate: ', name='alpha', type=FLOAT, value = "0.5", validate = check_LR)
    ])

    #replace ^(xor) with **(exponents) because in math ^ generally means the latter
    equation = info['equation'].replace('^', '**')
    
    #set the global variables to user input
    alpha = info['alpha']
    x_min = info['min']
    x_max = info['max']
    #compute the number of samples in x dimension we need for the graph to be detailed, capped at 500 (after it gets laggy)
    x_samples = min(500, 10 * (x_max - x_min))
    #make an array of all x_samples number of numbers between x_min and x_max
    x_vals = np.linspace(x_min, x_max, x_samples)

    #set the global variables to user input
    y_min = info['min']
    y_max = info['max']
    #compute the number of samples in y dimension we need for the graph to be detailed, capped at 500 (after it gets laggy)
    y_samples = min(500, 10 * (y_max - y_min))
    #make an array of all x_samples number of numbers between x_min and x_max
    y_vals = np.linspace(y_min, y_max, y_samples)

    #turn both of these 2D, computing the full meshgrid (initially with dummy values)
    x = np.outer(x_vals, np.ones(x_samples))
    y = np.outer(y_vals, np.ones(y_samples)).T
    
    #helper dictionary to get ready for computing full meshgrid
    var = {'x':x,'y':y}
    #fill the full meshgrid using eval() on the user-inputted function, with the meshgrid
    #previously computed as x and y, then fed into the function with the "var" dictionary
    z = eval(equation, globals(), var)
        
    # call getFig() to get all the stuff we needa plot
    res = getFig()
    #the graph is the first object
    fig = res[0]
    #this converts graph to html to enable it to be displayed
    html = fig.to_html(include_plotlyjs="require", full_html=False, default_height="800", default_width="800")
    #actually put it on there
    put_html(html)
    
    # put text saying minimum value and number of steps
    put_text(res[1]).style('font-size: 25px')
    put_text(res[2]).style('font-size: 25px')
    
    #link users can click to run again
    put_link(name="Run Again", url="https://gradientdescent.herokuapp.com/", new_window=False).style('font-size: 25px')

#set up website
if __name__ == '__main__':
    #create argumentparser to set up server
    parser = argparse.ArgumentParser()
    #add relevant arguments nessesary to set up the server (including the fact that "we are using port 8080")
    parser.add_argument("-p", "--port", type=int, default=8080)
    #extract the arguments 
    args = parser.parse_args()

    #start the server based on configured arguments
    pywebio.start_server(web, port=args.port)
    
    #comment the above line of code and uncomment the bottom line of code, then running this file will display the website on your local computer
    # web()