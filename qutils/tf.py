from scipy.optimize import least_squares
from scipy import signal
import matplotlib.pyplot as plt
import torch
import numpy as np

def estimateMIMOsys(model,device,IC,t,lookback=1):
   from nfoursid.nfoursid import NFourSID
   problemDim = len(IC)
   dt = t[1] - t[0]
   np.random.seed(0)
   u = np.random.randn(len(t), problemDim)  # Random input signal

   # 2. Collect output data from the RNN model
   model.eval()
   u_tensor = torch.tensor(u, dtype=torch.double).to(device)
   Y0 = torch.tensor(IC, dtype=torch.double).unsqueeze(0).to(device).reshape((1,1,problemDim))
   Y_pred = []

   with torch.no_grad():
      Y_t = Y0
      for i in range(len(u)):
         Y_t = model(Y_t)
         Y_pred.append(Y_t.cpu().numpy().squeeze())

   Y_pred = np.array(Y_pred)

   # 3. Prepare data for system identification
   u = u[lookback:]
   y = Y_pred[lookback:]

   # Prepare your input (u) and output (y) data
   # u: array of shape (samples, input_dim)
   # y: array of shape (samples, output_dim)

   # Create and fit the N4SID model
   n4sid_model = N4SID(n_states=problemDim)  # Number of states can be adjusted
   n4sid_model.fit(u, y)

   # Retrieve the estimated state-space matrices
   A_est = n4sid_model.A
   B_est = n4sid_model.B
   C_est = n4sid_model.C
   D_est = n4sid_model.D

   print(A_est)
   print(B_est)
   print(C_est)
   print(D_est)

   return

def estimateTranferFunction(model,device,IC,t,lookback=1):
   # Using ARX to obtain a transfer function from frequency response data from a recurrent model (in my case mamba networks)

   # 1. Generate input signal
   problemDim = len(IC)
   dt = t[1] - t[0]
   np.random.seed(0)
   u = np.random.randn(len(t), problemDim)  # Random input signal

   # 2. Collect output data from the RNN model
   model.eval()
   u_tensor = torch.tensor(u, dtype=torch.double).to(device)
   Y0 = torch.tensor(IC, dtype=torch.double).unsqueeze(0).to(device).reshape((1,1,problemDim))
   Y_pred = []

   with torch.no_grad():
      Y_t = Y0
      for i in range(len(u)):
         Y_t = model(Y_t)
         Y_pred.append(Y_t.cpu().numpy().squeeze())

   Y_pred = np.array(Y_pred)

   # 3. Prepare data for system identification
   u_sysid = u[lookback:]
   y_sysid = Y_pred[lookback:]



   # 4. Define model orders
   na = 2  # Order of the autoregressive part
   nb = 2  # Order of the input
   nk = 1  # Input delay

   # 5. Define ARX model and error function
   def arx_model(params, u, y, na, nb):
      a = params[:na]
      b = params[na:na+nb]
      y_pred = np.zeros_like(y)
      for t in range(max(na, nb), len(y)):
         y_past = y_pred[t-na:t][::-1]
         u_past = u[t-nk-nb+1:t-nk+1][::-1]
         y_pred[t] = -np.dot(a, y_past) + np.dot(b, u_past)
      return y_pred

   def error_function(params, u, y, na, nb):
      y_pred = arx_model(params, u, y, na, nb)
      error = y[max(na, nb):] - y_pred[max(na, nb):]
      return error.flatten()

   # 6. Perform parameter estimation
   initial_params = np.zeros(na + nb)
   params_estimated = []
   for dim in range(problemDim):
      res = least_squares(
         error_function,
         initial_params,
         args=(u_sysid[:, dim], y_sysid[:, dim], na, nb),
         verbose=1
      )
      params_estimated.append(res.x)

   params_estimated = np.array(params_estimated)

   # 7. Construct the transfer function
   transfer_functions = []
   for dim in range(problemDim):
      a_estimated = params_estimated[dim][:na]
      b_estimated = params_estimated[dim][na:na+nb]

      num = b_estimated
      den = np.hstack(([1], a_estimated))

      system = signal.dlti(num, den, dt=dt)
      transfer_functions.append(system)

   # 8. Validate the estimated model
   u_val = u_sysid
   y_val_true = y_sysid

   y_val_pred = []
   for dim in range(problemDim):
      system = transfer_functions[dim]
      _, y_out = signal.dlsim(system, u_val[:, dim])
      y_val_pred.append(y_out.squeeze())

   y_val_pred = np.array(y_val_pred).T

   # Calculate validation error
   validation_error = y_val_true - y_val_pred

   # Plot validation results
   for dim in range(problemDim):
      plt.figure(figsize=(10, 4))
      plt.plot(y_val_true[:, dim], label='Mamba Output')
      plt.plot(y_val_pred[:, dim], label='TF Output', linestyle='--')
      plt.title(f'Dimension {dim+1} Validation')
      plt.xlabel('Time Steps')
      plt.ylabel('Output')
      plt.legend()

   # 9. Print the estimated transfer functions
   # print(f"Validation Error: {validation_error}")
   for dim in range(problemDim):
      system = transfer_functions[dim]
      print(f"Estimated Transfer Function for Dimension {dim+1}:")
      print(f"Numerator coefficients: {system.num}")
      print(f"Denominator coefficients: {system.den}")
      print("\n")
