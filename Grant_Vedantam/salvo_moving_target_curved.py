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
import time

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


class TargetMotionCurved(om.ExplicitComponent):
    """
    Curved target kinematics: constant-speed, constant-turn-rate motion in the local
    tangent plane, mapped to (theta_T, phi_T) rates on a spherical Earth.

    - Heading evolves as: psi_T(t) = psi_T0 + omega * t, omega = V / R_turn.
    - Horizontal speed magnitude is constant: V (m/s).
    - Motion is expressed in local North/East components:
        vN = V * cos(psi_T)
        vE = -V * sin(psi_T)   (negative gives a left turn from North toward West when omega>0)
    - Latitude/longitude rates:
        phi_dot   = vN / (Re + h_T)
        theta_dot = vE / ((Re + h_T) * cos(phi_T))

    Bounded motion:
      If t > t_stop, the target stops moving (rates go to 0), which provides an
      upper bound on how far the target can travel.

    Altitude:
      Linearly increases from h0 to h0 + h_climb_total over [0, t_stop] and then holds.
      h_dot = h_climb_total / t_stop for t <= t_stop else 0.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('earth_radius', default=6_371_000.0)

    def setup(self):
        nn = self.options['num_nodes']

        # Dymos supplies phase time if the ODE has an input named 'time'
        self.add_input('time', val=np.zeros(nn), units='s')

        # Current target latitude and altitude (states)
        self.add_input('phi_T', val=np.zeros(nn), units='rad')
        self.add_input('h_T',   val=np.zeros(nn), units='m')

        # Parameters (constants over the phase; provided via Dymos parameters)
        self.add_input('target_speed', val=np.ones(nn)*100.0, units='m/s')
        self.add_input('turn_radius',  val=np.ones(nn)*20_000.0, units='m')
        self.add_input('psi_T0',       val=np.zeros(nn), units='rad')
        self.add_input('t_stop',       val=np.ones(nn)*250.0, units='s')
        self.add_input('h_climb_total', val=np.ones(nn)*1000.0, units='m')

        # Outputs: target state rates
        self.add_output('theta_T_dot', val=np.zeros(nn), units='rad/s')
        self.add_output('phi_T_dot',   val=np.zeros(nn), units='rad/s')
        self.add_output('h_T_dot',     val=np.zeros(nn), units='m/s')

        # Helpful for plotting/debugging (optional)
        self.add_output('psi_T', val=np.zeros(nn), units='rad')

        # Derivatives: FD is fine for now
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        Re = self.options['earth_radius']

        t = inputs['time'].reshape(-1)
        phi = inputs['phi_T'].reshape(-1)
        h   = inputs['h_T'].reshape(-1)

        V   = inputs['target_speed'].reshape(-1)
        R   = inputs['turn_radius'].reshape(-1)
        psi0 = inputs['psi_T0'].reshape(-1)
        tstop = inputs['t_stop'].reshape(-1)
        htot = inputs['h_climb_total'].reshape(-1)

        # Effective time for bounded motion/climb
        teff = np.minimum(t, tstop)

        omega = V / R  # rad/s
        psi = psi0 + omega * teff

        # Local velocities (North/East). Choose sign so omega>0 turns from North toward West.
        vN = V * np.cos(psi)
        vE = -V * np.sin(psi)

        # Stop motion after t_stop (upper bound on target travel)
        moving = (t <= tstop).astype(float)
        vN = vN * moving
        vE = vE * moving

        r = Re + h
        outputs['phi_T_dot'] = vN / r
        outputs['theta_T_dot'] = vE / (r * np.cos(phi) + 1e-12)

        # Linear climb over [0, t_stop], then hold
        # If t_stop is (near) zero, avoid divide-by-zero
        hdot = np.where(tstop > 1e-9, htot / tstop, 0.0)
        outputs['h_T_dot'] = hdot * moving

        outputs['psi_T'] = psi

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

        # Target motion model (for moving terminal target)
        self.add_subsystem('target_motion',
                           subsys=TargetMotionCurved(num_nodes=nn, earth_radius=6_371_000.0),
                           promotes_inputs=['time', 'phi_T', 'h_T', 'target_speed', 'turn_radius', 'psi_T0', 't_stop', 'h_climb_total'],
                           promotes_outputs=['theta_T_dot', 'phi_T_dot', 'h_T_dot', 'psi_T'])
        
        # Terminal error signals for boundary constraints: vehicle - target
        self.add_subsystem('target_err',
                           om.ExecComp(['theta_err = theta - theta_T',
                                        'phi_err   = phi   - phi_T',
                                        'h_err     = h     - h_T'],
                                       theta=np.zeros(nn), theta_T=np.zeros(nn),
                                       phi=np.zeros(nn),   phi_T=np.zeros(nn),
                                       h=np.zeros(nn),     h_T=np.zeros(nn),
                                       theta_err=np.zeros(nn), phi_err=np.zeros(nn), h_err=np.zeros(nn)),
                           promotes_inputs=['theta', 'theta_T', 'phi', 'phi_T', 'h', 'h_T'],
                           promotes_outputs=['theta_err', 'phi_err', 'h_err'])

def target_state_at_time(target, t, earth_radius=6_371_000.0):
    """Compute an approximate target (theta, phi, h) at time t (seconds) for initial guesses/plotting.

    Uses the same curved-motion model as TargetMotionCurved, but integrated analytically in a local
    North/East tangent plane, then mapped to small-angle changes in latitude/longitude.

    target dict keys (all SI/rad):
      theta0, phi0, h0
      speed_mps, turn_radius_m, psi0
      t_stop, h_climb_total_m
    """
    th0 = float(target['theta0'])
    ph0 = float(target['phi0'])
    h0  = float(target['h0'])
    V   = float(target['speed_mps'])
    R   = float(target['turn_radius_m'])
    psi0 = float(target['psi0'])
    tstop = float(target['t_stop'])
    htot  = float(target['h_climb_total_m'])

    teff = min(float(t), tstop)

    # Heading over time
    omega = V / R if R > 1e-9 else 0.0
    ang = omega * teff

    # Integrate local NE displacement for constant-speed, constant-turn-rate motion
    # Starting heading psi0, turning left (toward West) with vE = -V*sin(psi).
    # For psi(t) = psi0 + omega t:
    #   N(t) = (V/omega)[sin(psi0+omega t) - sin(psi0)]
    #   E(t) = (V/omega)[cos(psi0+omega t) - cos(psi0)] * (-1)?? handled by vE definition below
    if abs(omega) < 1e-12:
        # Straight line
        N = V * teff * np.cos(psi0)
        E = -V * teff * np.sin(psi0)
    else:
        psi1 = psi0 + ang
        N = (V/omega) * (np.sin(psi1) - np.sin(psi0))
        # With vE = -V*sin(psi), integral gives:
        E = (V/omega) * (np.cos(psi1) - np.cos(psi0))

    # Map to lat/long increments (small-angle approx)
    r = earth_radius + h0
    dphi = N / r
    dtheta = E / (r * np.cos(ph0) + 1e-12)

    # Altitude linear climb over [0, t_stop]
    if tstop > 1e-9:
        h = h0 + htot * (teff / tstop)
    else:
        h = h0

    return th0 + dtheta, ph0 + dphi, h

def solve_vehicle(initial_phi, initial_psi, final_gamma, final_psi,
                  target, t_duration=250.0, label="Case"):
    """
    Build, solve, and simulate one vehicle trajectory, returning arrays needed for plotting.
    Only final boundary values differ between cases; initial conditions are shared.

    Parameters
    ----------
    target : dict
        Target definition with keys:
          'theta0', 'phi0', 'h0' (initial location, rad/rad/m)
          'speed_mps', 'turn_radius_m', 'psi0', 't_stop', 'h_climb_total_m' (curved motion params)
    final_gamma, final_psi : float (rad)
        Desired terminal flight-path angle and heading.
    t_duration : float (s)
        Phase duration guess (free to optimize if you keep duration free).
    label : str
        Series label for plots.

    Returns
    -------
    out : dict
        { 'label': str, 'theta': deg, 'phi': deg, 'h': m, 'time': s, 'sol': Case, 'sim': Case }
    """
    t0 = time.time()

    #Build the problem
    p = om.Problem(model=om.Group())
    _, optimizer = set_pyoptsparse_opt('IPOPT', fallback=False)
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = optimizer
    p.model.linear_solver = om.DirectSolver(rhs_checking=True)
    p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    p.model.nonlinear_solver.options['iprint'] = 0

    p.driver.options['debug_print'] = []
    p.driver.options['print_results'] = False


    # (Optional) IPOPT options
    if optimizer == 'IPOPT':
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['linear_solver'] = 'mumps'
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['mu_strategy'] = 'adaptive'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['mu_init'] = 0.1
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['constr_viol_tol'] = 1e-5
        p.driver.opt_settings['compl_inf_tol'] = 1e-5
        p.driver.opt_settings['tol'] = 1e-5

    traj = p.model.add_subsystem('traj', dm.Trajectory())
    if optimizer == 'IPOPT':
        phase0 = traj.add_phase(
                                'phase0',
                                dm.Phase(
                                    ode_class=VehicleODE,
                                    transcription=dm.Radau(num_segments=10)
                                )
                            )
    else:
        phase0 = traj.add_phase(
                                'phase0',
                                dm.Phase(
                                    ode_class=VehicleODE,
                                    transcription=dm.Radau(num_segments=15, order=3)
                                )
                            )

    # Time/state/control defs
    phase0.set_time_options(fix_initial=True, fix_duration=False, units='s',
                        duration_ref=150.0, duration_bounds=(100.0, 500.0))

    phase0.add_state('h',     fix_initial=True, fix_final=False,  units='m',  rate_source='hdot',     lower=0, ref=40000, defect_ref=4.0e4)
    phase0.add_state('theta', fix_initial=True, fix_final=False,  units='rad', rate_source='thetadot', lower=0, upper=np.radians(6), defect_ref=np.radians(0.5))
    phase0.add_state('phi',   fix_initial=True, fix_final=False,  units='rad', rate_source='phidot',   lower=-np.radians(5), upper=np.radians(5), defect_ref=np.radians(0.5))

    # --- Moving target states (Method A): evolve target inside the phase ---
    phase0.add_state('theta_T', fix_initial=True, fix_final=False, units='rad', rate_source='theta_T_dot',
                 lower=-np.radians(180), upper=np.radians(180), defect_ref=np.radians(0.5))
    phase0.add_state('phi_T',   fix_initial=True, fix_final=False, units='rad', rate_source='phi_T_dot',
                 lower=-np.radians(90), upper=np.radians(90), defect_ref=np.radians(0.5))
    phase0.add_state('h_T',     fix_initial=True, fix_final=False, units='m',   rate_source='h_T_dot',
                 lower=-1.0e3, upper=2.0e5, defect_ref=4.0e4)
    # Curved target motion parameters (promoted into the ODE)
    phase0.add_parameter('target_speed', opt=False, val=target['speed_mps'], units='m/s')
    phase0.add_parameter('turn_radius',  opt=False, val=target['turn_radius_m'], units='m')
    phase0.add_parameter('psi_T0',       opt=False, val=target['psi0'], units='rad')
    phase0.add_parameter('t_stop',       opt=False, val=target['t_stop'], units='s')
    phase0.add_parameter('h_climb_total', opt=False, val=target['h_climb_total_m'], units='m')


    # Enforce vehicle hits the moving target at final time (vehicle - target = 0 at tf)
    phase0.add_boundary_constraint('theta_err', loc='final', equals=0.0, units='rad')
    phase0.add_boundary_constraint('phi_err',   loc='final', equals=0.0, units='rad')
    phase0.add_boundary_constraint('h_err',     loc='final', equals=0.0, units='m')

    phase0.add_state('v',     fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',     lower=0, ref=2000, defect_ref=2000)
    phase0.add_state('gamma', fix_initial=True, fix_final=False,  units='rad', rate_source='gammadot', lower=-np.radians(89), upper=np.radians(89), defect_ref=np.radians(1))
    phase0.add_state('psi',   fix_initial=True, fix_final=False,  units='rad', rate_source='psidot',   lower=-np.radians(90), upper=np.radians(90), defect_ref=np.radians(1))
    phase0.add_control('sigma', units='rad', opt=True, lower=np.radians(-165), upper=np.radians(165), rate_continuity=True)
    phase0.add_control('alpha', units='rad', opt=True, lower=np.radians(0), upper=np.radians(40), rate_continuity=True)

    # Objective Function
    phase0.add_objective('v', loc='final', ref=-0.1)

    p.setup(check=True)

    # Initial and terminal boundary values (share initial, vary final)
    phase0.set_time_val(initial=0.0, duration=t_duration, units='s')
    # Initial guesses (vehicle and target)
    # We build consistent end guesses for the moving target from the same curved-motion model.
    thetaT_end, phiT_end, hT_end = target_state_at_time(target, t_duration, earth_radius=6_371_000.0)
    
    # Vehicle state initial guesses (same structure as before; final values are just guesses)
    phase0.set_state_val('h',     [40000.0, 0.0], units='m')
    phase0.set_state_val('theta', [0.0, thetaT_end], units='rad')
    phase0.set_state_val('phi',   [initial_phi, phiT_end], units='rad')
    phase0.set_state_val('v',     [2000.0,  800.0], units='m/s')   # final free but initialized
    phase0.set_state_val('gamma', [0.0,     final_gamma], units='rad')
    phase0.set_state_val('psi',   [initial_psi, final_psi], units='rad')
    
    # Target initial condition at t=0 and a consistent end guess at t=t_duration.
    phase0.set_state_val('theta_T', [target['theta0'], thetaT_end], units='rad')
    phase0.set_state_val('phi_T',   [target['phi0'],   phiT_end],   units='rad')
    phase0.set_state_val('h_T',     [target['h0'],     hT_end],     units='m')


    # Control initial guesses
    phase0.set_control_val('sigma', [np.radians(0.0), np.radians(0.0)])
    phase0.set_control_val('alpha', [np.radians(0.0), np.radians(0.0)])

    # --- Solve and simulate ---
    dm.run_problem(p, simulate=True)
    t1 = time.time()

    print(f"[{label}] Optimization time: {t1 - t0:.2f} s")

    sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
    sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

    # Pull out arrays we’ll need for overlay plots (deg for angles)
    theta_sim = np.degrees(sim.get_val('traj.phase0.timeseries.theta')).flatten()
    phi_sim   = np.degrees(sim.get_val('traj.phase0.timeseries.phi')).flatten()
    h_sim     = sim.get_val('traj.phase0.timeseries.h').flatten()
    t_sim     = sim.get_val('traj.phase0.timeseries.time').flatten()

    # Final time to reach target
    t_final = t_sim[-1]
    print(f"[{label}] Final time to reach target: {t_final:.3f} s")


    return {
        'label': label,
        'theta': theta_sim,
        'phi':   phi_sim,
        'h':     h_sim,
        'time':  t_sim,
        't_final': t_final,
        'sol':   sol,
        'sim':   sim,
    }

def ts(case, var):
    """Return (t, y) from the simulation timeseries for a var name like 'alpha' or 'sigma'."""
    sim = case['sim']
    t = sim.get_val('traj.phase0.timeseries.time').ravel()
    y = sim.get_val(f'traj.phase0.timeseries.{var}').ravel()
    return t, y


# === Shared moving target definition (used by both Vehicle A and B) ===
EARTH_RADIUS_M = 6_371_000.0

# Initial target location (radians, meters)
TARGET_THETA0 = np.radians(1.5)   # initial downrange (theta) [rad]
TARGET_PHI0   = np.radians(1.0)   # initial crossrange (phi) [rad]
TARGET_H0     = 0.0               # initial altitude [m]

# Curved target motion parameters
TARGET_SPEED_MPS      = 60.0       # constant horizontal speed [m/s]
TARGET_TURN_RADIUS_M  = 20_000.0    # constant turn radius [m]
TARGET_PSI0_RAD       = 5.0         # initial heading [rad]; 0 = north, turning left moves toward west
TARGET_TSTOP_S        = 400.0       # upper bound on how long the target moves [s] (then it stops)
TARGET_CLIMB_TOTAL_M  = 1000.0      # altitude climbs linearly by this amount over [0, t_stop]

TARGET = dict(theta0=TARGET_THETA0,
              phi0=TARGET_PHI0,
              h0=TARGET_H0,
              speed_mps=TARGET_SPEED_MPS,
              turn_radius_m=TARGET_TURN_RADIUS_M,
              psi0=TARGET_PSI0_RAD,
              t_stop=TARGET_TSTOP_S,
              h_climb_total_m=TARGET_CLIMB_TOTAL_M)

# Define two terminal-condition variants (same target for both)
case_A = solve_vehicle(initial_phi=np.radians(2.0),
                       initial_psi=-np.radians(10.0),
                       final_gamma=-np.radians(45),
                       final_psi=-np.radians(85),
                       target=TARGET,
                       label="Vehicle A")

case_B = solve_vehicle(initial_phi=np.radians(0.0),
                       initial_psi=np.radians(10.0),
                       final_gamma=-np.radians(45),
                       final_psi=np.radians(45),
                       target=TARGET,
                       label="Vehicle B")

# --- Target trajectory from the simulation timeseries (same for both cases) ---
T_sim = case_A['sim']
tT = T_sim.get_val('traj.phase0.timeseries.time').ravel()
thetaT_deg = np.degrees(T_sim.get_val('traj.phase0.timeseries.theta_T')).ravel()
phiT_deg   = np.degrees(T_sim.get_val('traj.phase0.timeseries.phi_T')).ravel()
hT_m       = T_sim.get_val('traj.phase0.timeseries.h_T').ravel()

# Helpers to grab control histories
tA, alphaA = ts(case_A, 'alpha')
tB, alphaB = ts(case_B, 'alpha')
tA_s, sigmaA = ts(case_A, 'sigma')
tB_s, sigmaB = ts(case_B, 'sigma')

alphaA_deg = np.degrees(alphaA)
alphaB_deg = np.degrees(alphaB)
sigmaA_deg = np.degrees(sigmaA)
sigmaB_deg = np.degrees(sigmaB)

print("\n=== Impact Times ===")
print(f"Vehicle A impact time: {case_A['t_final']:.3f} s")
print(f"Vehicle B impact time: {case_B['t_final']:.3f} s")
print(f"Δt (B - A): {case_B['t_final'] - case_A['t_final']:.6f} s")

# Plot α vs time
plt.figure(figsize=(8,5))
plt.plot(tA, alphaA_deg, label='Vehicle A')
plt.plot(tB, alphaB_deg, label='Vehicle B')
plt.xlabel('Time (s)')
plt.ylabel('Angle of Attack α (deg)')
plt.title('Control History: α vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot σ vs time
plt.figure(figsize=(8,5))
plt.plot(tA_s, sigmaA_deg, label='Vehicle A')
plt.plot(tB_s, sigmaB_deg, label='Vehicle B')
plt.xlabel('Time (s)')
plt.ylabel('Bank Angle σ (deg)')
plt.title('Control History: σ vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot Top Down View (include the moving target path + start/end markers)
plt.figure(figsize=(8,6))
plt.plot(case_A['theta'], case_A['phi'], '-', label=case_A['label'])
plt.plot(case_B['theta'], case_B['phi'], '-', label=case_B['label'])
plt.plot(thetaT_deg, phiT_deg, '--', label='Target path')
plt.plot(thetaT_deg[0],  phiT_deg[0],  'o', label='Target start')
plt.plot(thetaT_deg[-1], phiT_deg[-1], 'x', label='Target end')
plt.xlabel('Downrange (deg)')
plt.ylabel('Crossrange (deg)')
plt.title('Ground Track (Crossrange vs Downrange)')
plt.grid(True)
plt.axis('equal')
plt.legend()

# Plot Side View (Altitude vs Downrange, plus target altitude if desired)
plt.figure(figsize=(8,6))
plt.plot(case_A['theta'], case_A['h'], '-', label=case_A['label'])
plt.plot(case_B['theta'], case_B['h'], '-', label=case_B['label'])
plt.plot(thetaT_deg, hT_m, '--', label='Target path (h)')
plt.xlabel('Downrange (deg)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs Downrange')
plt.grid(True)
plt.legend()

# 3D Plotly overlay: vehicles + target trajectory (line) + target start/end markers
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=case_A['theta'], y=case_A['phi'], z=case_A['h'],
    mode='lines', name=case_A['label'], line=dict(width=5)
))
fig.add_trace(go.Scatter3d(
    x=case_B['theta'], y=case_B['phi'], z=case_B['h'],
    mode='lines', name=case_B['label'], line=dict(width=5)
))

fig.add_trace(go.Scatter3d(
    x=thetaT_deg, y=phiT_deg, z=hT_m,
    mode='lines', name='Target path', line=dict(width=4, dash='dash')
))
fig.add_trace(go.Scatter3d(
    x=[thetaT_deg[0]], y=[phiT_deg[0]], z=[hT_m[0]],
    mode='markers', name='Target start', marker=dict(size=6)
))
fig.add_trace(go.Scatter3d(
    x=[thetaT_deg[-1]], y=[phiT_deg[-1]], z=[hT_m[-1]],
    mode='markers', name='Target end', marker=dict(size=6, symbol='x')
))

# Axes styling
xmin = min(case_A['theta'].min(), case_B['theta'].min(), thetaT_deg.min())
xmax = max(case_A['theta'].max(), case_B['theta'].max(), thetaT_deg.max())
ymin = min(case_A['phi'].min(),   case_B['phi'].min(),   phiT_deg.min())
ymax = max(case_A['phi'].max(),   case_B['phi'].max(),   phiT_deg.max())

fig.update_layout(
    scene=dict(
        xaxis_title='Downrange (deg)',
        yaxis_title='Crossrange (deg)',
        zaxis_title='Altitude (m)',
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(range=[ymin, ymax]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    title="3D Trajectory Overlay (Vehicles + Moving Target)",
    margin=dict(l=0, r=0, b=0, t=40)
)

pio.renderers.default = 'browser'
fig.show()
plt.show()
