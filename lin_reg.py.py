
#importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
#Function to compute the coefficients of linear regression
def coeff(x,y):
    n=np.size(x)
    m_x,m_y=np.mean(x),np.mean(y)
    ss_xy=np.sum(y*x-n*m_x*m_y)
    ss_xx=np.sum(x*x-n*m_x*m_y)
    b1=ss_xy/ss_xx
    b0=m_y-b1*m_x
    return(b0,b1)
#Our X and Y arrays
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
#Coefficients stored in list b
b=coeff(x,y)
print("The estimated coefficients are {} and {}".format(b[0],b[1]))
#plotting the regression curve
plt.scatter(x,y)
y_pred= b[0] + b[1]*x
plt.plot(x,y_pred)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()