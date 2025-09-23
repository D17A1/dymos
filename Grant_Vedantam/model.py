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

class Atmosphere(om.ExplicitComponent):
    """
    Exponential atmospheric density model 
    SI units
    Vedantam, Akella, Grant (2022)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('h',     val=np.ones(nn), desc='altitude',               units='m')
        self.add_output('rho',  val=np.ones(nn), desc='atmospheric density',    units='kg/m**3')
        
        arange = np.arange(nn, dtype=int)
        self.declare_partials('rho', 'h', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        h = inputs['h']
        H = 7500.0     # scale height in meters
        rho_0 = 1.2    # sea-level density in kg/m^3
        outputs['rho'] = rho_0 * np.exp(-h / H)

    def compute_partials(self, inputs, partials):
        h = inputs['h']
        H = 7500.0
        rho_0 = 1.2
        partials['rho', 'h'] = -rho_0 / H * np.exp(-h / H)

class Aerodynamics(om.ExplicitComponent):
    """
    Aerodynamic model for Vedantam & Grant trajectory problem.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', val=np.ones(nn), units='rad', desc='angle of attack')
        self.add_input('v',     val=np.ones(nn), units='m/s', desc='velocity')
        self.add_input('rho',   val=np.ones(nn), units='kg/m**3', desc='atmospheric density')

        self.add_output('drag', val=np.ones(nn), units='N', desc='drag force')
        self.add_output('lift', val=np.ones(nn), units='N', desc='lift force')

        arange = np.arange(nn)
        self.declare_partials('drag', ['alpha', 'v', 'rho'], rows=arange, cols=arange)
        self.declare_partials('lift', ['alpha', 'v', 'rho'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        CL_alpha = 1.5658
        CD_alpha2 = 1.6537
        CD_0 = 0.0612
        A_ref = 0.2919

        alpha = inputs['alpha']
        v = inputs['v']
        rho = inputs['rho']

        CL = CL_alpha * alpha
        CD = CD_alpha2 * alpha**2 + CD_0

        outputs['lift'] = 0.5 * rho * v**2 * CL * A_ref
        outputs['drag'] = 0.5 * rho * v**2 * CD * A_ref

    def compute_partials(self, inputs, J):
        CL_alpha = 1.5658
        CD_alpha2 = 1.6537
        CD_0 = 0.0612
        A_ref = 0.2919

        alpha = inputs['alpha']
        v = inputs['v']
        rho = inputs['rho']

        CL = CL_alpha * alpha
        CD = CD_alpha2 * alpha**2 + CD_0

        # Lift derivatives
        J['lift', 'alpha'] = 0.5 * rho * v**2 * A_ref * CL_alpha
        J['lift', 'v'] = rho * v * A_ref * CL
        J['lift', 'rho'] = 0.5 * v**2 * CL * A_ref

        # Drag derivatives
        dCD_dalpha = 2 * CD_alpha2 * alpha
        J['drag', 'alpha'] = 0.5 * rho * v**2 * A_ref * dCD_dalpha
        J['drag', 'v'] = rho * v * A_ref * CD
        J['drag', 'rho'] = 0.5 * v**2 * CD * A_ref

class DebrisDistance(om.ExplicitComponent):
    """
    Finds distance from vehicle to debris field center
    Converts from angle to meters
    Distance found using pythagorean
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('phi', shape=(nn,), units='rad')     # latitude
        self.add_input('theta', shape=(nn,), units='rad')   # longitude
        self.add_input('h', shape=(nn,), units='m')         # altitude

        self.add_output('dist_to_debris', shape=(nn,), units='m')

        ar = np.arange(nn)
        self.declare_partials('dist_to_debris', 'phi', rows=ar, cols=ar)
        self.declare_partials('dist_to_debris', 'theta', rows=ar, cols=ar)
        self.declare_partials('dist_to_debris', 'h', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        R_e = 6371000.0
        phi = inputs['phi']
        theta = inputs['theta']
        h = inputs['h']

        dphi = phi - phi0
        dtheta = theta - theta0
        dh = h - h0

        # Convert to meters
        dphi = dphi * R_e
        dtheta = dtheta * R_e

        outputs['dist_to_debris'] = np.sqrt(dphi**2 + dtheta**2 + dh**2)  # Euclidean approximation

    def compute_partials(self, inputs, partials):
        R_e = 6371000.0
        phi = inputs['phi']
        theta = inputs['theta']
        h = inputs['h']

        dphi = phi - phi0
        dtheta = theta - theta0
        dh = h - h0

        dphi = dphi * R_e
        dtheta = dtheta * R_e
        denom = np.sqrt(dphi**2 + dtheta**2 + dh**2)

        # Avoid division by zero
        denom = np.where(denom < 1e-8, 1e-8, denom)

        partials['dist_to_debris', 'phi'] = dphi / denom
        partials['dist_to_debris', 'theta'] = dtheta / denom
        partials['dist_to_debris', 'h'] = dh / denom

class FlightDynamics(om.ExplicitComponent):
    """
    Defines the dynamics of the vehicle
    3-DOF
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
    
        # Inputs
        self.add_input('h',     val=np.ones(nn), desc='altitude', units='m')
        self.add_input('theta', val=np.ones(nn), desc='longitude', units='rad')
        self.add_input('phi',   val=np.ones(nn), desc='latitude', units='rad')
        self.add_input('v',     val=np.ones(nn), desc='velocity', units='m/s')
        self.add_input('gamma', val=np.ones(nn), desc='flight path angle', units='rad')
        self.add_input('psi',   val=np.ones(nn), desc='heading angle', units='rad')
        self.add_input('sigma', val=np.ones(nn), desc='bank angle', units='rad')
        self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='rad')
        self.add_input('lift',  val=np.ones(nn), desc='lift force', units='N')
        self.add_input('drag',  val=np.ones(nn), desc='drag force', units='N')
    
        # Outputs
        self.add_output('hdot',     val=np.ones(nn), desc='altitude rate', units='m/s')
        self.add_output('thetadot', val=np.ones(nn), desc='longitude rate', units='rad/s')
        self.add_output('phidot',   val=np.ones(nn), desc='latitude rate', units='rad/s')
        self.add_output('vdot',     val=np.ones(nn), desc='velocity rate', units='m/s**2')
        self.add_output('gammadot', val=np.ones(nn), desc='flight path angle rate', units='rad/s')
        self.add_output('psidot',   val=np.ones(nn), desc='heading angle rate', units='rad/s')
    
        # Derivative structure
        partial_range = np.arange(nn, dtype=int)
    
        self.declare_partials('hdot',   'v',        rows=partial_range, cols=partial_range)
        self.declare_partials('hdot',   'gamma',    rows=partial_range, cols=partial_range)
    
        self.declare_partials('thetadot', 'v',      rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'gamma',  rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'psi',    rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'h',      rows=partial_range, cols=partial_range)
        self.declare_partials('thetadot', 'phi',    rows=partial_range, cols=partial_range)
    
        self.declare_partials('phidot', 'v',        rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'gamma',    rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'psi',      rows=partial_range, cols=partial_range)
        self.declare_partials('phidot', 'h',        rows=partial_range, cols=partial_range)
    
        self.declare_partials('vdot', 'drag',       rows=partial_range, cols=partial_range)
        self.declare_partials('vdot', 'gamma',      rows=partial_range, cols=partial_range)
        self.declare_partials('vdot', 'h',          rows=partial_range, cols=partial_range)
    
        self.declare_partials('gammadot', 'lift',   rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'sigma',  rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'gamma',  rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'h',      rows=partial_range, cols=partial_range)
        self.declare_partials('gammadot', 'v',      rows=partial_range, cols=partial_range)
    
        self.declare_partials('psidot', 'lift',     rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'sigma',    rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'gamma',    rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'v',        rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'phi',      rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'psi',      rows=partial_range, cols=partial_range)
        self.declare_partials('psidot', 'h',        rows=partial_range, cols=partial_range)


    def compute(self, inputs, outputs):
    
        # Have initial and terminal conditions
        h = inputs['h']                 # altitude
        theta = inputs['theta']         # longitude
        phi = inputs['phi']             # latitude
        v = inputs['v']                 # velocity
        gamma = inputs['gamma']         # flight path angle
        psi = inputs['psi']             # heading angle
    
        # Control inputs
        sigma = inputs['sigma']         # bank angle
        alpha = inputs['alpha']         # angle of attack
    
        # Aero forces
        lift = inputs['lift']
        drag = inputs['drag']

        # Constants
        mu = 3.986e14                   # gravitational constant [m^3/s^2]
        m = 340.1943                    # mass of vehicle [kg]
        A_ref = 0.2919                  # reference area (used for L/D if needed)
    
        r = 6378000 + h                 # total radial distance from Earth's center in meters
    
        # Needed trig terms
        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)
        sin_psi = np.sin(psi)  
        cos_psi = np.cos(psi)
        cos_phi = np.cos(phi)
        tan_phi = np.tan(phi)
        sin_sigma = np.sin(sigma)
        cos_sigma = np.cos(sigma)
    
        # Equations of motion from Vedantam Grant
        outputs['hdot'] = v * sin_gamma
        outputs['thetadot'] = v * cos_gamma * cos_psi / (r * cos_phi)   # longitude
        outputs['phidot'] = v * cos_gamma * sin_psi / r                 # latitude
        outputs['vdot'] = -drag / m - mu * sin_gamma / r**2
        outputs['gammadot'] = (lift * cos_sigma) / (m * v) - (mu * cos_gamma) / (v * r**2) + v / r * cos_gamma
        outputs['psidot'] = (lift * sin_sigma) / (m * v * cos_gamma) - v / r * cos_gamma * cos_psi * tan_phi

    def compute_partials(self, inputs, J):
        # Have initial and terminal conditions
        h = inputs['h']                 # altitude
        theta = inputs['theta']         # longitude
        phi = inputs['phi']             # latitude
        v = inputs['v']                 # velocity
        gamma = inputs['gamma']         # flight path angle
        psi = inputs['psi']             # heading angle
    
        # Control inputs
        sigma = inputs['sigma']         # bank angle
        alpha = inputs['alpha']         # angle of attack
    
        # Aero forces
        lift = inputs['lift']
        drag = inputs['drag']

        # Constants
        mu = 3.986e14                   # gravitational parameter [m^3/s^2]
        m = 340.1943                    # mass of vehicle [kg]
        A_ref = 0.2919                  # reference area (used for L/D if needed)
    
        r = 6378000 + h  
    
        # Needed trig terms
        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)

        sin_psi = np.sin(psi)  
        cos_psi = np.cos(psi)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        tan_phi = np.tan(phi)
        sec_phi = 1 / np.cos(phi)

        sin_sigma = np.sin(sigma)
        cos_sigma = np.cos(sigma)

        sec_phi_sq = 1 / np.cos(phi)**2

        # Partial derivatives of eqn 1a
        J['hdot', 'v'] = sin_gamma
        J['hdot', 'gamma'] = v * cos_gamma

        # Partial derivatives of eqn 1b
        J['thetadot', 'v'] = cos_gamma * cos_psi / (r * cos_phi)
        J['thetadot', 'gamma'] = -v * sin_gamma * cos_psi / (r * cos_phi)
        J['thetadot', 'psi'] = -v * cos_gamma * sin_psi / (r * cos_phi)
        J['thetadot', 'h'] = -v * cos_gamma * cos_psi / (r**2 * cos_phi)
        J['thetadot', 'phi'] = v * cos_gamma * cos_psi * sin_phi / (r * cos_phi**2)

        # Partial derivatives of eqn 1c
        J['phidot', 'v'] = cos_gamma * sin_psi / r
        J['phidot', 'gamma'] = -v * sin_gamma * sin_psi / r
        J['phidot', 'psi'] = v * cos_gamma * cos_psi / r
        J['phidot', 'h'] = -v * cos_gamma * sin_psi / (r ** 2)

        # Partial derivatives of eqn 1d
        J['vdot', 'drag']  = -1.0 / m
        J['vdot', 'gamma'] = -mu * cos_gamma / r**2
        J['vdot', 'h']     =  2.0 * mu * sin_gamma / r**3

        # Partial derivatives of eqn 1e
        J['gammadot', 'lift'] = cos_sigma / (m * v)
        J['gammadot', 'sigma'] = -lift * sin_sigma / (m * v)
        J['gammadot', 'v'] = -lift * cos_sigma / (m * v**2) + mu * cos_gamma / (v**2 * r**2) + cos_gamma / r
        J['gammadot', 'gamma'] = mu * sin_gamma / (v * r**2) - v * sin_gamma / r
        J['gammadot', 'h'] = 2 * mu * cos_gamma / (v * r**3) - v * cos_gamma / (r**2)

        # Check for zeros
        #if np.any(np.abs(lift) < 1e-8):
        #    print("Zero lift at node(s):", np.where(np.abs(lift) < 1e-8))

        # Partial derivatives of eqn 1f
        J['psidot', 'lift'] = sin_sigma / (m * v * cos_gamma)
        J['psidot', 'sigma'] = lift * cos_sigma / (m * v * cos_gamma)
        J['psidot', 'v'] = -lift * sin_sigma / (m * v**2 * cos_gamma) - cos_gamma * cos_psi * tan_phi / r
        J['psidot', 'gamma'] = lift * sin_sigma * sin_gamma / (m * v * cos_gamma**2) + v * sin_gamma * cos_psi * tan_phi / r
        J['psidot', 'psi'] = v * cos_gamma * sin_psi * tan_phi / r
        J['psidot', 'phi'] = -v * cos_gamma * cos_psi * sec_phi_sq / r
        J['psidot', 'h'] = v * cos_gamma * cos_psi * tan_phi / r**2

class VehicleODE(Group):
    """
    The ODE for the Shuttle reentry problem following Vedantam & Grant (2022).
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Atmosphere model: maps altitude to density
        self.add_subsystem('atmosphere',
                           subsys=Atmosphere(num_nodes=nn),
                           promotes_inputs=['h'],
                           promotes_outputs=['rho'])

        # Aerodynamics model: maps angle of attack, velocity, and density to lift and drag
        self.add_subsystem('aerodynamics',
                           subsys=Aerodynamics(num_nodes=nn),
                           promotes_inputs=['alpha', 'v', 'rho'],
                           promotes_outputs=['lift', 'drag'])

        # Debris avoidance constraint
        self.add_subsystem('debris_dist',
                           subsys=DebrisDistance(num_nodes=nn),
                           promotes_inputs=['phi', 'theta', 'h'],
                           promotes_outputs=['dist_to_debris'])

        # Dynamics model: 6-DOF planar dynamics using Vedantam/Grant equations
        self.add_subsystem('eom',
                           subsys=FlightDynamics(num_nodes=nn),
                           promotes_inputs=[
                               'h', 'theta', 'phi', 'v', 'gamma', 'psi',
                               'sigma','alpha', 'lift', 'drag'
                           ],
                           promotes_outputs=[
                               'hdot', 'thetadot', 'phidot', 'vdot',
                               'gammadot', 'psidot'
                           ])

import numpy as np
import matplotlib.pyplot as plt

def CirclePlot(center_x, center_y, radius_m, view='topdown', earth_radius=6371000.0, **plot_kwargs):
    """
    Generate x and y coordinates for a circle for plotting with plt.plot().

    Parameters
    ----------
    center_x : float
        Center x-coordinate (radians).
    center_y : float
        Center y-coordinate (radians for 'topdown', meters for 'lateral').
    radius_m : float
        Radius of the circle in meters.
    view : str
        'topdown' (crossrange vs downrange in radians) or 'lateral' (altitude vs downrange).
    earth_radius : float
        Radius of Earth in meters (default: 6371000).
    **plot_kwargs : dict
        Additional keyword arguments passed to plt.plot().

    Returns
    -------
    None
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    
    if view == 'topdown':
        # Convert radius to radians for both axes
        dx = (radius_m / earth_radius) * np.cos(theta)
        dy = (radius_m / earth_radius) * np.sin(theta)
        x = center_x + dx
        y = center_y + dy
    elif view == 'lateral':
        # x is radians, y is meters
        dx = (radius_m / earth_radius) * np.cos(theta)
        dy = radius_m * np.sin(theta)
        x = center_x + dx
        y = center_y + dy
    else:
        raise ValueError("view must be either 'topdown' or 'lateral'")

    plt.plot(x, y, **plot_kwargs)



# Create the OpenMDAO problem
p = om.Problem(model=om.Group())
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=False)
#_, optimizer = set_pyoptsparse_opt('SLSQP', fallback=False)

p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()

p.driver.options['optimizer'] = optimizer
p.driver.options['print_results'] = True


# IPOPT causing issues
if optimizer == 'IPOPT':
    p.driver.opt_settings['print_level'] = 0
    p.driver.opt_settings['mu_strategy'] = 'adaptive'
    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
    p.driver.opt_settings['mu_init'] = 0.1
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['constr_viol_tol'] = 1e-4
    p.driver.opt_settings['compl_inf_tol'] = 1e-4
    p.driver.opt_settings['tol'] = 1e-5

# Define the trajectory and add the phase
traj = p.model.add_subsystem('traj', dm.Trajectory())

phase0 = traj.add_phase('phase0',
    dm.Phase(ode_class=VehicleODE,
             transcription=dm.Radau(num_segments=10)))

# Add the phase to the model
p.model.linear_solver = om.DirectSolver()
p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

# Set time options, will be single phase (glide)
phase0.set_time_options(fix_initial=True, fix_duration=False, units='s')

# Define state variables
phase0.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='hdot',
                 lower=0, ref=40000)
phase0.add_state('theta', fix_initial=True, fix_final=True, units='rad', rate_source='thetadot',
                 lower=0, upper=np.radians(6))
phase0.add_state('phi', fix_initial=True, fix_final=True, units='rad', rate_source='phidot',
                 lower=-np.radians(3), upper=np.radians(3))
phase0.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
                 lower=0, ref=2000)
# Changed from free to fixed final
phase0.add_state('gamma', fix_initial=True, fix_final=True, units='rad', rate_source='gammadot',
                 lower=-np.radians(89), upper=np.radians(89))
phase0.add_state('psi', fix_initial=True, fix_final=True, units='rad', rate_source='psidot',
                 lower=-np.radians(180), upper=np.radians(180))

# Control input (bank angle sigma)
phase0.add_control('sigma', units='rad', opt=True,
                   lower=np.radians(-1), upper=np.radians(165))
phase0.add_control('alpha', units='rad', opt=True,
                   lower=np.radians(0), upper=np.radians(40))

# Enforce a minimum distance to stay outside the debris field
#phase0.add_path_constraint('dist_to_debris', lower=debris_radius)
phase0.add_timeseries_output('dist_to_debris', shape=(1,))

# Objective 1: maximize final theta (longitude) downrange distance
#phase0.add_objective('phi', loc='final', ref=-0.01)
# Objective 2: maximize final velocity
phase0.add_objective('v', loc='final', ref=-1.0)
# Objective 3: minimize time
#phase0.add_objective('t', loc='final', ref=1.0)

# Setup the problem
p.setup(check=True)

# Set initial and final time
phase0.set_time_val(initial=0.0, duration=2000, units='s')

# Boundary conditions from Vedantam & Grant Table 2
phase0.set_state_val('h', [40000.0, 0.0], units='m')
phase0.set_state_val('theta', [0.0, np.radians(3.5)], units='rad')
phase0.set_state_val('phi', [0.0, np.radians(0.1)], units='rad')
phase0.set_state_val('v', [2000.0, 1100.0], units='m/s')  # final free but initialized
phase0.set_state_val('gamma', [0.0, -np.radians(45)], units='rad')
phase0.set_state_val('psi', [0.0, np.radians(0)], units='rad')

# Bank angle control guess (sigma)
phase0.set_control_val('sigma', [np.radians(0.0), np.radians(0.0)])
phase0.set_control_val('alpha', [np.radians(0.0), np.radians(0.0)])

#p.check_partials(compact_print=True, method='cs')  # or 'fd'

# Run the problem
dm.run_problem(p, simulate=True)

sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

plot_results([
    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.sigma',
     'Time (s)', 'Bank Angle σ (rad)'),
    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
     'Time (s)', 'Angle of Attack α (rad)'),
    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.dist_to_debris',
               'time (s)', 'Distance (m)')
], title='Vedantam-Grant Controls', p_sol=sol, p_sim=sim)
plt.tight_layout()

# Get minimum distance to debris along entire flight
print(min(sol.get_val('traj.phase0.timeseries.dist_to_debris')))

# Plot state variables
plot_results([
    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.h',
     'Time (s)', 'Altitude h (m)'),

    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.v',
     'Time (s)', 'Velocity v (m/s)'),

    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.gamma',
     'Time (s)', 'Flight Path Angle γ (rad)'),

    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.phi',
     'Time (s)', 'Latitude φ (rad)'),

    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.psi',
     'Time (s)', 'Heading Angle ψ (rad)')
], title='Vedantam-Grant State', p_sol=sol, p_sim=sim)

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
CirclePlot(center_x=np.degrees(theta0), center_y=np.degrees(phi0), radius_m=debris_radius,
           view='topdown', color='red', linestyle='--', label='Debris Radius')
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
CirclePlot(center_x=np.degrees(theta0), center_y=h0, radius_m=debris_radius,
           view='lateral', color='red', linestyle='--', label='Debris Radius')
plt.xlabel('Downrange (deg)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs. Downrange')
plt.grid(True)
plt.tight_layout()

# Create 3D plot of trajectory
x = theta_sim
y = phi_sim
z = altitude_sim

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='lines+markers',
    line=dict(width=4),
    marker=dict(size=2),
)])

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    title="3D Trajectory of Hypersonic Vehicle",
    margin=dict(l=0, r=0, b=0, t=40)
)

pio.renderers.default = 'browser'

fig.show()
plt.show()
