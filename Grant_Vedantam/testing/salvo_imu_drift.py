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

def apply_random_position_shift(state_dict, shift_m=250.0, rng=None):
    """
    Apply a random 3D position shift of fixed magnitude (meters) to the state.
    The shift is generated in local ENU coordinates and converted to:
      theta [rad], phi [rad], h [m]

    Parameters
    ----------
    state_dict : dict
        Must contain keys 'theta', 'phi', 'h'.
    shift_m : float
        Magnitude of the position shift in meters.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    shifted_state : dict
        Copy of state_dict with perturbed theta, phi, h.
    shift_info : dict
        ENU components and converted angular shifts for debugging/printing.
    """
    if rng is None:
        rng = np.random.default_rng()

    shifted_state = dict(state_dict)

    # Random 3D unit vector
    vec = rng.normal(size=3)
    vec /= np.linalg.norm(vec)

    # ENU components (meters)
    dE = shift_m * vec[0]
    dN = shift_m * vec[1]
    dU = shift_m * vec[2]

    Re = 6_371_000.0
    phi = shifted_state['phi']
    h = shifted_state['h']
    r = Re + h

    # Convert local ENU position offsets to your state variables
    dphi = dN / r
    dtheta = dE / (r * np.cos(phi) + 1e-12)
    dh = dU

    shifted_state['phi'] += dphi
    shifted_state['theta'] += dtheta
    shifted_state['h'] += dh

    shift_info = {
        'dE_m': dE,
        'dN_m': dN,
        'dU_m': dU,
        'dtheta_rad': dtheta,
        'dphi_rad': dphi,
        'dh_m': dh,
        'mag_m': shift_m,
    }

    return shifted_state, shift_info

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

def solve_vehicle(initial_phi, initial_psi, final_theta, final_phi, final_gamma, final_psi,
                  t_duration=250.0, label="Case", initial_state_override=None):
    """
    Build, solve, and simulate one vehicle trajectory, returning arrays needed for plotting.
    Only final boundary values differ between cases; initial conditions are shared.

    Parameters
    ----------
    final_theta, final_phi : float (rad)
        Desired terminal longitude/latitude.
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
                        duration_ref=250.0, duration_bounds=(182.0, 500.0))

    phase0.add_state('h',     fix_initial=True, fix_final=True,  units='m',  rate_source='hdot',     lower=0, ref=40000, defect_ref=4.0e4)
    phase0.add_state('theta', fix_initial=True, fix_final=True,  units='rad', rate_source='thetadot', lower=0, upper=np.radians(6), defect_ref=np.radians(0.5))
    phase0.add_state('phi',   fix_initial=True, fix_final=True,  units='rad', rate_source='phidot',   lower=-np.radians(5), upper=np.radians(5), defect_ref=np.radians(0.5))
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

    # Initials (optionally overridden for re-planning)
    if initial_state_override is None:
        h0     = 40000.0
        theta0 = 0.0
        phi0   = initial_phi
        v0     = 2000.0
        gamma0 = 0.0
        psi0   = initial_psi
    else:
        h0     = float(initial_state_override['h'])
        theta0 = float(initial_state_override['theta'])
        phi0   = float(initial_state_override['phi'])
        v0     = float(initial_state_override['v'])
        gamma0 = float(initial_state_override['gamma'])
        psi0   = float(initial_state_override['psi'])

    phase0.set_state_val('h',     [h0,     0.0],         units='m')
    phase0.set_state_val('theta', [theta0, final_theta], units='rad')
    phase0.set_state_val('phi',   [phi0,   final_phi],   units='rad')
    phase0.set_state_val('v',     [v0,     800.0],       units='m/s')   # final free but initialized
    phase0.set_state_val('gamma', [gamma0, final_gamma], units='rad')
    phase0.set_state_val('psi',   [psi0,   final_psi],   units='rad')

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


def get_state_at_time(sim_case, t_query, prefix='traj.phase0.timeseries.'):
    """Interpolate the vehicle state at an absolute time (seconds) from a Dymos simulation Case."""
    t = sim_case.get_val(prefix + 'time').ravel()

    def interp(var):
        y = sim_case.get_val(prefix + var).ravel()
        return float(np.interp(t_query, t, y))

    return {
        'h':     interp('h'),
        'theta': interp('theta'),
        'phi':   interp('phi'),
        'v':     interp('v'),
        'gamma': interp('gamma'),
        'psi':   interp('psi'),
    }

def ts(case, var):
    """Return (t, y) from the simulation timeseries for a var name like 'alpha' or 'sigma'."""
    sim = case['sim']
    t = sim.get_val('traj.phase0.timeseries.time').ravel()
    y = sim.get_val(f'traj.phase0.timeseries.{var}').ravel()
    return t, y

###############################################################################
# Two-stage (re-planning / MPC-like)
#   Stage 1: plan to a guessed target
#   At t = 50 s, "reveal" a new target 0.1 deg further downrange
#   Stage 2: re-plan from the stage-1 state at t=50 s to the revealed target
###############################################################################

t_reveal = 50.0  # seconds
rng = np.random.default_rng(42)   # fixed seed for reproducibility
GPS_SHIFT_M = 250.0

# Shared mission target definition
TARGET_THETA_TRUE_DEG = 2.0
TARGET_PHI_TRUE_DEG   = 1.0
TARGET_THETA_GUESS_DEG = TARGET_THETA_TRUE_DEG + 0.01  # revealed +0.1 deg downrange
TARGET_PHI_GUESS_DEG = TARGET_PHI_TRUE_DEG + 0.05

target_theta_guess = np.radians(TARGET_THETA_GUESS_DEG)
target_theta_true  = np.radians(TARGET_THETA_TRUE_DEG)
target_phi_guess    = np.radians(TARGET_PHI_GUESS_DEG)
target_phi_true    = np.radians(TARGET_PHI_TRUE_DEG)


def concat_timeseries(case_stage1, case_stage2, var, t_cut, offset):
    """Concatenate a timeseries variable from two simulation cases."""
    sim1 = case_stage1['sim']
    sim2 = case_stage2['sim']
    t1 = sim1.get_val('traj.phase0.timeseries.time').ravel()
    y1 = sim1.get_val(f'traj.phase0.timeseries.{var}').ravel()
    m1 = t1 <= t_cut
    t1 = t1[m1]
    y1 = y1[m1]

    t2 = sim2.get_val('traj.phase0.timeseries.time').ravel() + offset
    y2 = sim2.get_val(f'traj.phase0.timeseries.{var}').ravel()

    return np.concatenate([t1, t2]), np.concatenate([y1, y2])


# --- Vehicle A, Stage 1 (guessed target) ---
case_A1 = solve_vehicle(initial_phi=np.radians(1.7),
                        initial_psi=-np.radians(0.0),
                        final_theta=target_theta_guess,
                        final_phi=target_phi_guess,
                        final_gamma=-np.radians(45),
                        final_psi=0.0,  # keep terminal psi unconstraining-ish in stage 1
                        label="Vehicle A (stage 1)")

A1_state_nominal = get_state_at_time(case_A1['sim'], t_reveal)
A1_state, A1_shift = apply_random_position_shift(A1_state_nominal, shift_m=GPS_SHIFT_M, rng=rng)

print("\n[Vehicle A] Stage-2 GPS update shift:")
print(f"  dE = {A1_shift['dE_m']:.3f} m, dN = {A1_shift['dN_m']:.3f} m, dU = {A1_shift['dU_m']:.3f} m")
print(f"  dtheta = {np.degrees(A1_shift['dtheta_rad']):.8f} deg, "
      f"dphi = {np.degrees(A1_shift['dphi_rad']):.8f} deg, "
      f"dh = {A1_shift['dh_m']:.3f} m")

# --- Vehicle A, Stage 2 (revealed target) ---
remaining_guess_A = max(182.0, case_A1['t_final'] - t_reveal)
case_A2 = solve_vehicle(initial_phi=np.radians(0.0),
                        initial_psi=np.radians(0.0),
                        final_theta=target_theta_true,
                        final_phi=target_phi_true,
                        final_gamma=-np.radians(45),
                        final_psi=0.0,
                        t_duration=remaining_guess_A,
                        label="Vehicle A (stage 2)",
                        initial_state_override=A1_state)


# --- Vehicle B, Stage 1 (guessed target) ---
earth_radius = 6_371_000.0
case_B1 = solve_vehicle(initial_phi=0.0/earth_radius,
                        initial_psi=np.radians(0.0),
                        final_theta=target_theta_guess,
                        final_phi=target_phi_guess,
                        final_gamma=-np.radians(45),
                        final_psi=0.0,
                        label="Vehicle B (stage 1)")

B1_state_nominal = get_state_at_time(case_B1['sim'], t_reveal)
B1_state, B1_shift = apply_random_position_shift(B1_state_nominal, shift_m=GPS_SHIFT_M, rng=rng)

print("\n[Vehicle B] Stage-2 GPS update shift:")
print(f"  dE = {B1_shift['dE_m']:.3f} m, dN = {B1_shift['dN_m']:.3f} m, dU = {B1_shift['dU_m']:.3f} m")
print(f"  dtheta = {np.degrees(B1_shift['dtheta_rad']):.8f} deg, "
      f"dphi = {np.degrees(B1_shift['dphi_rad']):.8f} deg, "
      f"dh = {B1_shift['dh_m']:.3f} m")

# --- Vehicle B, Stage 2 (revealed target) ---
remaining_guess_B = max(182.0, case_B1['t_final'] - t_reveal)
case_B2 = solve_vehicle(initial_phi=np.radians(0.0),
                        initial_psi=np.radians(0.0),
                        final_theta=target_theta_true,
                        final_phi=target_phi_true,
                        final_gamma=-np.radians(45),
                        final_psi=0.0,
                        t_duration=remaining_guess_B,
                        label="Vehicle B (stage 2)",
                        initial_state_override=B1_state)


# Build stitched outputs for plotting
def stitch(case1, case2, label):
    # stage-1 truncated to t_reveal, stage-2 shifted by +t_reveal
    t1 = case1['time']
    m1 = t1 <= t_reveal

    out = {
        'label': label,
        'time':  np.concatenate([case1['time'][m1], case2['time'] + t_reveal]),
        'theta': np.concatenate([case1['theta'][m1], case2['theta']]),
        'phi':   np.concatenate([case1['phi'][m1],   case2['phi']]),
        'h':     np.concatenate([case1['h'][m1],     case2['h']]),
        't_final': t_reveal + case2['t_final'],
        'sim1': case1['sim'],
        'sim2': case2['sim'],
    }
    return out


case_A = stitch(case_A1, case_A2, 'Vehicle A (two-stage)')
case_B = stitch(case_B1, case_B2, 'Vehicle B (two-stage)')


# Controls stitched
tA, alphaA = concat_timeseries(case_A1, case_A2, 'alpha', t_reveal, t_reveal)
tB, alphaB = concat_timeseries(case_B1, case_B2, 'alpha', t_reveal, t_reveal)

tA_s, sigmaA = concat_timeseries(case_A1, case_A2, 'sigma', t_reveal, t_reveal)
tB_s, sigmaB = concat_timeseries(case_B1, case_B2, 'sigma', t_reveal, t_reveal)

# Convert to degrees
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

# Plot Top Down View
plt.figure(figsize=(8,6))
plt.plot(case_A['theta'], case_A['phi'], '-',  label=case_A['label'])
plt.plot(case_B['theta'], case_B['phi'], '-',  label=case_B['label'])
plt.scatter(np.degrees(target_theta_guess),
            np.degrees(target_phi_guess),
            marker='o', s=120, c='red',
            label='Target (Initial Guess)')
plt.scatter(np.degrees(target_theta_true),
            np.degrees(target_phi_true),
            marker='*', s=200, c='black',
            label='Target (Revealed True)')
plt.xlabel('Downrange (deg)')
plt.ylabel('Crossrange (deg)')
plt.title('Ground Track (Crossrange vs Downrange)')
plt.grid(True); plt.axis('equal'); plt.legend()

# Plot Side View
plt.figure(figsize=(8,6))
plt.plot(case_A['theta'], case_A['h'], '-', label=case_A['label'])
plt.plot(case_B['theta'], case_B['h'], '-', label=case_B['label'])
plt.xlabel('Downrange (deg)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs Downrange')
plt.grid(True); plt.legend()

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
    x=[np.degrees(target_theta_guess)],
    y=[np.degrees(target_phi_guess)],
    z=[0.0],
    mode='markers',
    marker=dict(size=8, color='red'),
    name='Target (Initial Guess)'
))
fig.add_trace(go.Scatter3d(
    x=[np.degrees(target_theta_true)],
    y=[np.degrees(target_phi_true)],
    z=[0.0],
    mode='markers',
    marker=dict(size=10, color='black', symbol='diamond'),
    name='Target (Revealed True)'
))

# Axes styling
xmin = min(case_A['theta'].min(), case_B['theta'].min())
xmax = max(case_A['theta'].max(), case_B['theta'].max())
ymin = min(case_A['phi'].min(),   case_B['phi'].min())
ymax = max(case_A['phi'].max(),   case_B['phi'].max())

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
    title="3D Trajectory Overlay",
    margin=dict(l=0, r=0, b=0, t=40)
)

pio.renderers.default = 'browser'
fig.show()
plt.show()
