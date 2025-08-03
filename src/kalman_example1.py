import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

 #function to generate noisy measurements
def getMeasurement(updateNumber):
  if updateNumber == 1:
    getMeasurement.currentPosition = 0
    getMeasurement.currentVelocity = 60

  dt = 0.1 #timestep: ten measurements per second

  w = 8 * np.random.rand(1) #velocity noise
  v = 8 * np.random.rand(1) #positional noise

  z = getMeasurement.currentPosition + getMeasurement.currentVelocity * dt + v
  getMeasurement.currentPosition = z - v
  getMeasurement.currentVelocity = 60 + w
  return [z,getMeasurement.currentPosition, getMeasurement.currentVelocity]

def kfilter(z, updateNumber):
  dt = 0.1
  sd = 10

  if updateNumber == 1:
    kfilter.x = np.array([[0],
                          [20]])
    kfilter.P = np.array([[5,0],
                          [0,5]])
    kfilter.A = np.array([[1,dt],
                          [0,1]])
    kfilter.H = np.array([[1],
                          [0]])
    kfilter.HT = np.array([[1],
                           [0]])
    kfilter.R = 10
    kfilter.Q = sd * np.array([[dt**3/3,dt**2],[dt**2,dt]])

    #predict state forward
    x_prime = kfilter.A.dot(kfilter.x)
    #predict covariance forward
    P_prime = kfilter.A.dot(kfilter.P).dot(kfilter.A.T) + kfilter.Q
    
    #compute kalman gain
    S = kfilter.H.dot(P_prime).dot(kfilter.HT) + kfilter.R
    K = P_prime.dot(kfilter.HT) * (1/S)

    #estimate state
    residual = z - kfilter.H.dot(x_prime)
    kfilter.x = x_prime + K*residual

    #estimate covariance
    kfilter.P = P_prime - K.dot(kfilter.H).dot(P_prime)

    return [kfilter.x[0], kfilter.x[1], kfilter.P,K]
  
