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
    kfilter.x = np.array([[0],[20]])
    kfilter.P = np.array([[5,0],
                          [0,5]])
    kfilter.A = np.array([[1,dt],
                          [0,1]])
    kfilter.H = np.array([[1,0]])

    kfilter.R = 1
    kfilter.Q = sd * np.array([[dt**3/3,dt**2],[dt**2,dt]])

  #predict state forward
  x_prime = kfilter.A.dot(kfilter.x)
  #predict covariance forward
  P_prime = kfilter.A.dot(kfilter.P).dot(kfilter.A.T) + kfilter.Q
    
  #compute kalman gain
  S = kfilter.H.dot(P_prime).dot(kfilter.H.T) + kfilter.R
  K = P_prime.dot(kfilter.H.T) * (1/S)

  #estimate state
  residual = z - kfilter.H.dot(x_prime)
  kfilter.x = x_prime + K*residual
  #estimate covariance
  kfilter.P = P_prime - K.dot(kfilter.H).dot(P_prime)
  return [kfilter.x[0], kfilter.x[1], kfilter.P,K]

def testFilter():  
  #define range of measurements to loop
  dt = 0.1
  t = np.linspace(0,10, num = 300)
  numOfMeasurements = len(t)

  #init arrys to save off data so it could be plotted
  measTime = []
  measPos = []
  measDifPos = []
  estDifPos = []
  estPos = []
  estVel = []
  posBound3Sigma = []
  posGain = []
  velGain = []

  #loop through each measurement
  for k in range(1,numOfMeasurements):
    #generate the latest measurements
    z=getMeasurement(k)
    #call filter and return new state
    f = kfilter(z[0],k)
    #save off that state so that it could be plotted 
    measTime.append(k)
    measPos.append(z[0])
    measDifPos.append(z[0]-z[1])
    estDifPos.append(f[0]-z[1])
    estPos.append(f[0])
    estVel.append(f[1])
    posVar = f[2]
    posBound3Sigma.append(3*np.sqrt(posVar[0][0]))
    K = f[3]
    posGain.append(K[0][0])
    velGain.append(K[1][0])

  return [measTime,measPos,estPos,estVel,measDifPos,estDifPos,posBound3Sigma,posGain,velGain];

# Execute Test Filter Function
t = testFilter()
# Plot Results
plot1 = plt.figure(1)
plt.scatter(t[0], t[1])
plt.plot(t[0], t[2])
plt.ylabel('Position')
plt.xlabel('Time')
plt.grid(True)

plot2 = plt.figure(2)
plt.plot(t[0], t[3])
plt.ylabel('Velocity (meters/seconds)')
plt.xlabel('Update Number')
plt.title('Velocity Estimate On Each Measurement Update \n', fontweight="bold")
plt.legend(['Estimate'])
plt.grid(True)

plot3 = plt.figure(3)
plt.scatter(t[0], t[4], color = 'red')
plt.plot(t[0], t[5])
plt.legend(['Estimate', 'Measurement'])
plt.title('Position Errors On Each Measurement Update \n', fontweight="bold")
plt.ylabel('Position Error (meters)')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0, 300])

plot4 = plt.figure(4)
plt.plot(t[0], t[7])
plt.plot(t[0], t[8])
plt.ylabel('Gain')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0, 100])
plt.legend(['Position Gain', 'Velocity Gain'])
plt.title('Position and Velocity Gains \n', fontweight="bold")
plt.show()