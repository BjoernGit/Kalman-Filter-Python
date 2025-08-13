#!/usr/bin/env python3
# Listing 6.x – Extended Kalman Filter Demo (rewritten with English comments)

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


# ---------- Listing 6.2: EKF Function ----------
def ekfilter(z, updateNumber):   # z = [m_r, m_b, m_cov, t_r, t_b, t_time, t_x, t_y, m_x, m_y]
    dt = 0.5
    j = updateNumber

    # ---------- Initialize State ----------
    if updateNumber == 0:  # First update: derive position only from measurement
        # Compute position values from measurements (polar -> cartesian)
        temp_x = z[0][j] * np.sin(z[1][j] * np.pi / 180.0)   # x = r*sin(b)
        temp_y = z[0][j] * np.cos(z[1][j] * np.pi / 180.0)   # y = r*cos(b)

        # State vector: [x, y, xv, yv]^T – positions set, velocity = 0
        ekfilter.x = np.array([[temp_x],
                               [temp_y],
                               [0.0],
                               [0.0]])

        # State covariance – set to zero for the first step
        ekfilter.P = np.array([[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0]])

        # State transition (constant velocity)
        ekfilter.A = np.array([[1.0, 0.0, dt, 0.0],
                               [0.0, 1.0, 0.0, dt],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])

        # Measurement covariance and process noise
        ekfilter.R = z[2][j]  # 2x2 matrix from measurement function
        ekfilter.Q = np.array([[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0]])

        # Placeholders to keep return format consistent
        residual = np.array([[0.0, 0.0],
                             [0.0, 0.0]])
        K = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 0.0]])

        return [ekfilter.x[0][0], ekfilter.x[1][0], ekfilter.P,
                ekfilter.x[2][0], ekfilter.x[3][0], K, residual]

    # ---------- Second Update: estimate velocity from two positions ----------
    if updateNumber == 1:
        # previous estimate
        prev_x = ekfilter.x[0][0]
        prev_y = ekfilter.x[1][0]

        # current position from measurement
        temp_x = z[0][j] * np.sin(z[1][j] * np.pi / 180.0)
        temp_y = z[0][j] * np.cos(z[1][j] * np.pi / 180.0)

        # velocity (pos2 - pos1) / dt
        temp_xv = (temp_x - prev_x) / dt
        temp_yv = (temp_y - prev_y) / dt

        # new state (position + computed velocity)
        ekfilter.x = np.array([[temp_x],
                               [temp_y],
                               [temp_xv],
                               [temp_yv]])

        # large initial uncertainty
        ekfilter.P = np.array([[100.0, 0.0,   0.0,   0.0],
                               [0.0,  100.0,  0.0,   0.0],
                               [0.0,    0.0, 250.0,  0.0],
                               [0.0,    0.0,   0.0, 250.0]])

        # State transition
        ekfilter.A = np.array([[1.0, 0.0, dt, 0.0],
                               [0.0, 1.0, 0.0, dt],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])

        # Measurement noise and system noise
        ekfilter.R = z[2][j]
        sd = 1.0  # spectral density
        ekfilter.Q = np.array([[(sd * dt**3) / 3.0, 0.0,                 (sd * dt**2) / 2.0, 0.0],
                               [0.0,                 (sd * dt**3) / 3.0, 0.0,                 (sd * dt**2) / 2.0],
                               [(sd * dt**2) / 2.0,  0.0,                 sd * dt,            0.0],
                               [0.0,                 (sd * dt**2) / 2.0,  0.0,                 sd * dt]])

        residual = np.array([[0.0, 0.0],
                             [0.0, 0.0]])
        K = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 0.0]])

        return [ekfilter.x[0][0], ekfilter.x[1][0], ekfilter.P,
                ekfilter.x[2][0], ekfilter.x[3][0], K, residual]

    # ---------- Third Update and beyond: prediction + EKF correction ----------
    # Predict state and covariance forward
    x_prime = ekfilter.A.dot(ekfilter.x)
    P_prime = ekfilter.A.dot(ekfilter.P).dot(ekfilter.A.T) + ekfilter.Q

    # Form state-to-measurement transition (Jacobian) for h(x)
    x1 = x_prime[0][0]
    y1 = x_prime[1][0]
    x_sq = x1 * x1
    y_sq = y1 * y1
    den = x_sq + y_sq
    den1 = np.sqrt(den)

    # H – Jacobian of [r, b] with respect to [x, y, xv, yv]
    ekfilter.H = np.array([[x1 / den1,      y1 / den1,      0.0, 0.0],
                           [y1 / den,      -x1 / den,       0.0, 0.0]])
    ekfilter.HT = ekfilter.H.T

    # Measurement covariance
    ekfilter.R = z[2][j]

    # Compute Kalman Gain
    S = ekfilter.H.dot(P_prime).dot(ekfilter.HT) + ekfilter.R
    K = P_prime.dot(ekfilter.HT).dot(inv(S))

    # Measurement vector
    temp_z = np.array([[z[0][j]],
                       [z[1][j]]])

    # Convert predicted cartesian state to polar range & azimuth
    pred_x = x_prime[0][0]
    pred_y = x_prime[1][0]
    sumSquares = pred_x * pred_x + pred_y * pred_y
    pred_r = np.sqrt(sumSquares)
    pred_b = np.arctan2(pred_x, pred_y) * 180.0 / np.pi

    h_small = np.array([[pred_r],
                        [pred_b]])

    # Residual (Innovation): measurement minus prediction
    residual = temp_z - h_small

    # Update state vector
    ekfilter.x = x_prime + K.dot(residual)

    # Update covariance
    ekfilter.P = P_prime - K.dot(ekfilter.H).dot(P_prime)

    return [ekfilter.x[0][0], ekfilter.x[1][0], ekfilter.P,
            ekfilter.x[2][0], ekfilter.x[3][0], K, residual]


# ---------- Listing 6.3: Computing Measurements Function ----------
def getMeasurements():
    # Measurements are taken 2 times per second
    t = np.linspace(0.0, 50.0, num=100)
    numOfMeasurements = len(t)

    # Define x and y initial points -> Range ≈ 4100 m
    x = 2900.0
    y = 2900.0
    vel = 22.0  # m/s (≈ 50 mph)

    # Storage for "true" positions/times
    t_time, t_x, t_y, t_r, t_b = [], [], [], [], []

    # Compute trajectory in cartesian and convert to polar
    for i in range(numOfMeasurements):
        dt = 0.5
        t_time.append(t[i])

        # Motion: only in x direction
        x = x + dt * vel
        y = y

        t_x.append(x)
        t_y.append(y)

        temp = x * x + y * y
        r = np.sqrt(temp)
        t_r.append(r)

        # Azimuth in degrees (note: arctan2(x, y) to match original book code)
        b = np.arctan2(x, y) * 180.0 / np.pi
        t_b.append(b)

    # Measurement data with noise
    m_r, m_b, m_cov = [], [], []
    m_x, m_y = [], []

    # Bearing standard deviation = 9 mrad (in degrees)
    sig_b = 0.009 * 180.0 / np.pi
    # Range standard deviation = 10 m
    sig_r = 10.0

    for ii in range(0, len(t_time)):
        # Random error for each measurement
        temp_sig_b = sig_b * np.random.randn()
        temp_sig_r = sig_r * np.random.randn()

        # Measured values = true + error
        temp_b = t_b[ii] + temp_sig_b
        temp_r = t_r[ii] + temp_sig_r

        # Measurement covariance matrix (2x2)
        m_cov.append(np.array([[sig_r * sig_r, 0.0],
                               [0.0,           sig_b * sig_b]]))

        # Store polar measurements
        m_b.append(temp_b)
        m_r.append(temp_r)

        # (for analysis) cartesian points derived from polar measurements
        m_x.append(temp_r * np.sin(temp_b * np.pi / 180.0))
        m_y.append(temp_r * np.cos(temp_b * np.pi / 180.0))

    return [m_r, m_b, m_cov, t_r, t_b, t_time, t_x, t_y, m_x, m_y]


# ---------- Listing 6.4: Simulation & Visualization ----------
if __name__ == "__main__":
    f_x, f_y = [], []
    f_x_sig, f_y_sig = [], []
    f_xv, f_yv = [], []
    f_xv_sig, f_yv_sig = [], []

    z = getMeasurements()

    for iii in range(0, len(z[0])):
        f = ekfilter(z, iii)
        f_x.append(f[0])
        f_y.append(f[1])
        f_xv.append(f[3])
        f_yv.append(f[4])

        # Standard deviation from covariance matrix
        f_x_sig.append(np.sqrt(f[2][0][0]))
        f_y_sig.append(np.sqrt(f[2][1][1]))

    # Plot 1 – Range
    plt.figure(1)
    plt.grid(True)
    plt.plot(z[5], z[3])     # actual range
    plt.scatter(z[5], z[0])  # measured range
    plt.title('Actual Range vs Measured Range')
    plt.legend(['Ship Actual Range', 'Ship Measured Range'])
    plt.ylabel('Range (meters)')
    plt.xlabel('Seconds (s)')

    # Plot 2 – Azimuth
    plt.figure(2)
    plt.grid(True)
    plt.plot(z[5], z[4])     # actual azimuth
    plt.scatter(z[5], z[1])  # measured azimuth
    plt.title('Actual Azimuth vs Measured Azimuth')
    plt.legend(['Ship Actual Azimuth', 'Ship Measured Azimuth'])
    plt.ylabel('Azimuth (degrees)')
    plt.xlabel('Seconds (s)')

    # Plot 3 – Velocity estimates
    plt.figure(3)
    plt.grid(True)
    plt.plot(z[5], f_xv)
    plt.plot(z[5], f_yv)
    plt.title('Velocity Estimate On Each Measurement Update \n')
    plt.legend(['X Velocity Estimate', 'Y Velocity Estimate'])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Seconds (s)')

    # Compute position errors and 3-sigma bounds
    e_x_err, e_x_3sig, e_x_3sig_neg = [], [], []
    e_y_err, e_y_3sig, e_y_3sig_neg = [], [], []
    for m in range(0, len(z[0])):
        e_x_err.append(f_x[m] - z[6][m])
        e_x_3sig.append(3.0 * f_x_sig[m])
        e_x_3sig_neg.append(-3.0 * f_x_sig[m])

        e_y_err.append(f_y[m] - z[7][m])
        e_y_3sig.append(3.0 * f_y_sig[m])
        e_y_3sig_neg.append(-3.0 * f_y_sig[m])

    # Plot 4 – X error
    plt.figure(4)
    plt.grid(True)
    y1 = plt.scatter(z[5], e_x_err)
    y2, = plt.plot(z[5], e_x_3sig, color='green')
    y3, = plt.plot(z[5], e_x_3sig_neg, color='green')
    plt.ylabel('Position Error (meters)')
    plt.xlabel('Seconds (s)')
    plt.title('X Position Estimate Error Containment \n', fontweight='bold')
    plt.legend([y1, y2], ['X Position Error', '3 Sigma Error Bound'])

    # Plot 5 – Y error
    plt.figure(5)
    plt.grid(True)
    y1 = plt.scatter(z[5], e_y_err)
    y2, = plt.plot(z[5], e_y_3sig, color='green')
    y3, = plt.plot(z[5], e_y_3sig_neg, color='green')
    plt.ylabel('Position Error (meters)')
    plt.xlabel('Seconds (s)')
    plt.title('Y Position Estimate Error Containment \n', fontweight='bold')
    plt.legend([y1, y2], ['Y Position Error', '3 Sigma Error Bound'])

    plt.show()
