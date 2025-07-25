import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.api import Group
from dymos.examples.plotting import plot_results

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


class FlightDynamics(om.ExplicitComponent):
    """
    Defines the dynamics of the vehicle
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
        mu = 3.986e14                   # gravitational parameter [m^3/s^2]
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
    
        # Equations of motion from Vedantam & Grant
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
        if np.any(np.abs(lift) < 1e-8):
            print("Zero lift at node(s):", np.where(np.abs(lift) < 1e-8))

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

# Create the OpenMDAO problem
p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()
p.driver.options['optimizer'] = 'SLSQP'

# Define the trajectory and add the phase
traj = p.model.add_subsystem('traj', dm.Trajectory())

phase0 = traj.add_phase('phase0',
    dm.Phase(ode_class=VehicleODE,
             transcription=dm.Radau(num_segments=15, order=3)))

# Add the phase to the model
p.model.linear_solver = om.DirectSolver()
p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)

# Set time options
phase0.set_time_options(fix_initial=True, fix_duration=False, units='s')

# Define state variables
phase0.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='hdot',
                 lower=0, ref=40000)
phase0.add_state('theta', fix_initial=True, fix_final=True, units='rad', rate_source='thetadot',
                 lower=0, upper=np.radians(10))
phase0.add_state('phi', fix_initial=True, fix_final=True, units='rad', rate_source='phidot',
                 lower=0, upper=np.radians(5))
phase0.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
                 lower=0, ref=2000)
phase0.add_state('gamma', fix_initial=True, fix_final=False, units='rad', rate_source='gammadot',
                 lower=np.radians(-90), upper=np.radians(90))
phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad', rate_source='psidot',
                 lower=0, upper=2 * np.pi)

# Control input (bank angle sigma)
phase0.add_control('sigma', units='rad', opt=True,
                   lower=np.radians(-89), upper=np.radians(89))
phase0.add_control('alpha', units='rad', opt=True,
                   lower=np.radians(-89), upper=np.radians(89))

# Objective 1: maximize final theta (longitude) downrange distance
# phase0.add_objective('theta', loc='final', ref=-0.01)
# Objective 2: maximize final velocity
phase0.add_objective('v', loc='final', ref=-1.0)


# Setup the problem
p.setup(check=True)

# Set initial and final time
phase0.set_time_val(initial=0.0, duration=2000, units='s')

# Boundary conditions from Vedantam & Grant Table 2
phase0.set_state_val('h', [40000.0, 0.0], units='m')
phase0.set_state_val('theta', [0.0, np.radians(5.0)], units='rad')
phase0.set_state_val('phi', [0.0, np.radians(1.0)], units='rad')
phase0.set_state_val('v', [2000.0, 1500.0], units='m/s')  # final free but initialized
phase0.set_state_val('gamma', [0.0, -0.2], units='rad')
phase0.set_state_val('psi', [0.0, 0.1], units='rad')

# Bank angle control guess (sigma)
phase0.set_control_val('sigma', [np.radians(0.0), np.radians(0.0)])
phase0.set_control_val('alpha', [np.radians(0.0), np.radians(0.0)])

p.check_partials(compact_print=True, method='cs')  # or 'fd'

# Run the problem
dm.run_problem(p, simulate=True)

sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

plot_results([
    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.sigma',
     'Time (s)', 'Bank Angle σ (rad)'),

    ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
     'Time (s)', 'Angle of Attack α (rad)')
], title='Vedantam-Grant Controls', p_sol=sol, p_sim=sim)

plt.tight_layout()
plt.show()

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

plt.tight_layout()
plt.show()

