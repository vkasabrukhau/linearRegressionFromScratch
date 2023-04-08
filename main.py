from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64) #converting to numpy array, also setting it to default datatype, datatype doesn't matter here yet
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation = False): #hm is how many datapoints we want to create, variance is how variable we want the dataset to be, step is how far on average to step up the y value per point, correlation is positive, negative, or none (true or false).
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step #steps it up if its a positive correlation
        elif correlation and correlation == 'neg':
            val-=step #steps it down if its a negative correlation
    xs = [i for i in range(len(ys))] # basically just creates a list of xs that all linearly range from 1 to whatever the length of the ys is
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
# if variance is increased, the coefficient goes up


def best_fit_slope_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2) #squared error for the entire line

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig] #will make a single value, and each value is the mean of y, for each y in the original line
    squared_error_regr = squared_error(ys_orig, ys_line) #
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean) #this is where we actually fined the coefficient

xs, ys = create_dataset(40, 20, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs] #one line for loop that makes a list of y values for the xs via the regression line

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

predict_x = 8
predict_y = (m * predict_x) + b

plt.scatter(xs, ys) #plots regular xs and ys
plt.plot(xs, regression_line) #plots regression line
plt.scatter(predict_x, predict_y, s=100, color='g') #plots prediction
plt.show() #shows graph