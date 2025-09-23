import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.api import Group
from dymos.examples.plotting import plot_results
from openmdao.utils.general_utils import set_pyoptsparse_opt
import plotly.graph_objects as go
import plotly.io as pio

# Define Center of Debris Field
phi0 = np.radians(1.0)
theta0 = np.radians(2.35)
h0 = 23800.0
debris_radius = 10000.0

sol = om.CaseReader('./model_out/dymos_solution.db').get_case('final')
sim = om.CaseReader('./model_out/traj_simulation_0_out/dymos_simulation.db').get_case('final')

# Earth's radius in meters
R_e = 6371000.0

# Pull longitude and latitude from timeseries
theta_sol = sol.get_val('traj.phase0.timeseries.theta')  # longitude (rad)
phi_sol = sol.get_val('traj.phase0.timeseries.phi')      # latitude (rad)

theta_sim = sim.get_val('traj.phase0.timeseries.theta')
phi_sim = sim.get_val('traj.phase0.timeseries.phi')

"""
# Convert to meters
downrange_sol = R_e * theta_sol
crossrange_sol = R_e * phi_sol

downrange_sim = R_e * theta_sim
crossrange_sim = R_e * phi_sim
"""

# Convert to deg
downrange_sol = 180 * theta_sol / np.pi
crossrange_sol = 180 * phi_sol / np.pi

downrange_sim = 180 * theta_sim / np.pi
crossrange_sim = 180 * phi_sim / np.pi

# Plots top down view cross vs down range
plt.figure(figsize=(8,6))
plt.plot(downrange_sol, crossrange_sol, 'o', label='solution')   # km
plt.plot(downrange_sim, crossrange_sim, '-', label='simulation') # km
plt.xlabel('Downrange (deg)')
plt.ylabel('Crossrange (deg)')
plt.plot(sol.get_val('traj.phase0.timeseries.theta'),
         sol.get_val('traj.phase0.timeseries.phi'))
plt.scatter(np.degrees(theta0), np.degrees(phi0), c='r', label='Debris Field Center')
plt.title('Vehicle Ground Track (Crossrange vs Downrange)')
plt.grid(True)
plt.legend()
plt.axis('equal')  # preserve aspect ratio

# Extract altitude and velocity from the solution timeseries
altitude_sol = sol.get_val('traj.phase0.timeseries.h')  # in meters
velocity_sol = sol.get_val('traj.phase0.timeseries.v')  # in m/s

# Convert altitude to km and velocity to km/s
altitude_km = altitude_sol / 1000.0
velocity_kms = velocity_sol / 1000.0

"""
# Plots Velocity vs Alt
plt.figure(figsize=(8,6))
plt.plot(velocity_kms, altitude_km, 'b-')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Altitude (km)')
plt.title('Altitude vs. Velocity')
plt.grid(True)
plt.tight_layout()
"""

altitude_sim = sim.get_val('traj.phase0.timeseries.h')  # in meters

# Plots alt vs downrange
plt.figure(figsize=(8,6))
plt.plot(downrange_sol, altitude_sol, 'o', label='solution')   
plt.plot(downrange_sim, altitude_sim, '-', label='simulation') 
plt.scatter(np.degrees(theta0), h0, c='r', label='Debris Center')
plt.xlabel('Downrange (deg)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Downrange')
plt.grid(True)
plt.tight_layout()

# Create 3D plot of trajectory
x = theta_sol * R_e
y = phi_sol * R_e
z = altitude_sol

pio.renderers.default = 'browser'

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])

fig.update_layout(
    title="3D Trajectory of Hypersonic Vehicle",
)


fig.show()
#plt.show()
