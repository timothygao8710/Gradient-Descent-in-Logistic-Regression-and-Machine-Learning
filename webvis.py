import pywebio
from pywebio.input import input, FLOAT, input_group, NUMBER
from pywebio.output import put_text, put_html, put_markdown
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
import numpy as np 
from numpy import sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, log, gcd, sinc
# from math import sin, cos, tan, gcd, log10, sqrt, asin, acos, atan, dist, sinh, cosh, tanh, gamma, log
from math import ceil
from random import randrange

from flask import Flask, send_from_directory
import argparse

pywebio.config(theme="minty")

#initialize global variables
x, x_min, x_max, x_samples, x_vals = 0,0,0,0,0 
y, y_min, y_max, y_samples, y_vals = 0,0,0,0,0
equation, z = 0, 0
#Used for calculating learning rate
#Inversely proportional to number of steps
alpha = 0.015

#do cool functions (gcd, atan, dist)
#play bruh sound when bad
#hasn't been extensively tested

#calculate initial learning rate
#learning rate usually undergoes hyperparameter tuning for particular surface/graph, this is just for demonstration purposes
def calcLR():
    return 0.05 * alpha * (x_samples + y_samples)

#Function for changing learning rate
#a is some coefficient, can be tuned
#Other options include LR - a, LR/(1 + a * LR), LR ^ a
def updateLR(LR, a = 0.3):
    return LR * np.exp(-a)
    #LR over iterations graph becomes: e ^ (- x * a)

def check_range(val):
    if val > 50:
        return 'Please choose a value less than 50 for detailed graph'
    elif val < -50:
        return 'Please choose a value larger than -50 for detailed graph'

def check_LR(val):
    if val < 0 or val > 1:
        return "Please enter a percentage between 0 and 1"
    
def check_eq(eq_str):
    eq_str = eq_str.replace('^', '**')
    try:
        for x in range(-50, 50, 10):
            for y in range(-50, 50, 10):
                eval(eq_str)
    except Exception as e:
        return 'Please enter a valid expression: ' + str(e)

def draw_vectors(step_history):
    plotted = []
    
    x_locs, y_locs, z_locs = [], [], []
    x_delta, y_delta, z_delta = [], [], []
    
    result_dot = 0
    
    for i in range(len(step_history)):
        xi, yi = step_history[i][0], step_history[i][1]
        xn, yn = 0, 0
        
        if i != len(step_history)-1:
            xn, yn = step_history[i+1][0], step_history[i+1][1]
        else:
            xn, yn = xi, yi
        
        zi, zn = z[xi][yi], z[xn][yn]
        xi, yi, xn, yn = x_vals[xi], y_vals[yi], x_vals[xn], y_vals[yn]
        
        dx, dy, dz = xn - xi, yn - yi, zn - zi
        
        x_locs.append(xi)
        y_locs.append(yi)
        z_locs.append(zi)
        
        x_delta.append(dx)
        y_delta.append(dy)
        z_delta.append(dz)
        
        if i == len(step_history)-1:
            result_dot = (xi, yi, zi)
    
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
    
    plotted.append(cones)
    plotted.append(lines)
#     plotted.append(dot)
    return plotted


def gradient_descent(x_init, y_init, max_steps = 500):
    LR = ceil(calcLR())
    xi, yi = x_init, y_init
    res = []
    
    for i in range(max_steps):
        res.append((xi, yi))
        
        best_deriv = (0, 0, 0)
        
        for j in range(-LR, LR+1):
            for k in range(-LR, LR+1):
                nx, ny = xi + j, yi + k
                if nx < 0 or ny < 0 or nx >= x_samples or ny >= y_samples:
                    continue
                    
                cur_deriv = (z[nx][ny] - z[xi][yi], nx, ny)
                best_deriv = min(best_deriv, cur_deriv)
        
        if best_deriv[0] == 0:
            break
            
        xi = best_deriv[1]
        yi = best_deriv[2]

        LR = ceil(updateLR(LR))

    return (z[xi][yi], res)

def getFig():
    best_res = (np.max(z), [])
    for i in range(250):
        xi, yi = randrange(x_samples), randrange(y_samples)

        cur_res = gradient_descent(xi, yi)
        
        if cur_res[0] < best_res[0] or (len(cur_res[1]) > len(best_res[1])):
            best_res = cur_res
    
    plotted = draw_vectors(best_res[1])

    plotted.append(go.Surface(x=x, y=y, z=z, colorscale = 'viridis', opacity=0.6, showscale=False))

    fig = go.Figure(data=plotted)

    mv = best_res[1][len(best_res[1])-1]
    
    # camera = dict(
    #     eye=dict(x=x_vals[mv[0]] * 0.8, y=y_vals[mv[1]] * 0.8, z = np.max(z) + 0.05 * (np.max(z) - np.min(z)))
    # )

    fig.update_layout(title= "f(x,y) = " + equation.replace('**', '^'))

    return (
        fig,
        "Minimum value at: (" + str(round(x_vals[mv[0]], 4)) + ", " + str(round(y_vals[mv[1]], 4)) + ", " + str(round(best_res[0], 4)) + ")",
        "Converged in " + str(len(best_res[1])-1) + " steps with initial learning rate of " + str(alpha)
    )

def web():
    global x_min, x_max, x_samples, x_vals, y_min, y_max, y_samples, y_vals, x, y, z, alpha, equation
    info = input_group("User info",[
        input('f(x, y) = ', name='equation', value = "sin(x) ^ 10 + cos(10 + y * x) * cos(x)", validate = check_eq),
        input('Minimum value: ', name='min', type=FLOAT, value = "-3", validate = check_range),
        input('Maximum value: ', name='max', type=FLOAT, value = "3", validate = check_range),
        input('Learning rate: ', name='alpha', type=FLOAT, value = "0.5", validate = check_range)
    ])

    equation = info['equation'].replace('^', '**')
    alpha = info['alpha']
    x_min = info['min']
    x_max = info['max']
    x_samples = min(500, 10 * (x_max - x_min))
    x_vals = np.linspace(x_min, x_max, x_samples)

    y_min = info['min']
    y_max = info['max']
    y_samples = min(500, 10 * (y_max - y_min))
    y_vals = np.linspace(y_min, y_max, y_samples)

    x = np.outer(x_vals, np.ones(x_samples))
    y = np.outer(y_vals, np.ones(y_samples)).T
    
    var = {'x':x,'y':y}
    z = eval(equation, globals(), var)
        
    res = getFig()
    fig = res[0]
    html = fig.to_html(include_plotlyjs="require", full_html=False, default_height="800", default_width="800")
    put_html(html)
    
    put_text(res[1]).style('font-size: 25px')
    put_text(res[2]).style('font-size: 25px')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    pywebio.start_server(web, port=args.port)
    # web()