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
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

def print_vehicle_state_before_impact(case, dt_before=10.0):
    """
    Print stitched vehicle state dt_before seconds before impact.
    Works on the output of stitch(case1, case2, ...).
    """
    t = case['time'].ravel()
    t_final = case['t_final']
    t_query = t_final - dt_before

    if t_query < t[0]:
        raise ValueError(
            f"Requested time {dt_before} s before impact is before the start of the stitched trajectory."
        )

    def interp(var):
        y = np.asarray(case[var]).ravel()
        return float(np.interp(t_query, t, y))

    h = interp('h')
    theta = interp('theta')
    phi = interp('phi')
    v = interp('v')
    gamma = interp('gamma')
    psi = interp('psi')

    print(f"\n[{case['label']}] State {dt_before:.1f} s before impact")
    print(f"  mission time   = {t_query:.3f} s")
    print(f"  velocity       = {v:.3f} m/s")
    print(f"  altitude h     = {h:.3f} m")
    print(f"  downrange θ    = {np.degrees(theta):.6f} deg")
    print(f"  crossrange φ   = {np.degrees(phi):.6f} deg")
    print(f"  gamma          = {np.degrees(gamma):.6f} deg")
    print(f"  psi            = {np.degrees(psi):.6f} deg")

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



# =============================================================================
# Importable Dymos runner API
# =============================================================================

@dataclass
class VehicleInitialCondition:
    """Initial vehicle state used when starting a Dymos phase from launch."""
    h: float = 40000.0
    theta: float = 0.0
    phi: float = 0.0
    v: float = 2000.0
    gamma: float = 0.0
    psi: float = 0.0


@dataclass
class TargetCondition:
    """Terminal target position for a Dymos phase."""
    theta: float = np.radians(2.0)
    phi: float = np.radians(1.0)
    h: float = 0.0


@dataclass
class TerminalGuess:
    """Guesses for terminal states that are not hard-constrained."""
    v: float = 800.0
    gamma: float = -np.radians(45.0)
    psi: float = 0.0


@dataclass
class DymosSolverConfig:
    """
    Solver/transcription/bounds settings for the Dymos vehicle problem.

    Keep mission-specific parameters out of this object. This should describe
    how the problem is transcribed and solved, not where the target is.
    """
    optimizer: str = "IPOPT"
    fallback_optimizer: bool = False
    num_segments_ipopt: int = 10
    num_segments_other: int = 15
    order_other: int = 3

    duration_ref: float = 250.0
    duration_bounds: tuple[float, float] = (182.0, 500.0)

    theta_bounds: tuple[float, float] = (0.0, np.radians(6.0))
    phi_bounds: tuple[float, float] = (-np.radians(5.0), np.radians(5.0))
    gamma_bounds: tuple[float, float] = (-np.radians(89.0), np.radians(89.0))
    psi_bounds: tuple[float, float] = (-np.radians(90.0), np.radians(90.0))
    sigma_bounds: tuple[float, float] = (-np.radians(165.0), np.radians(165.0))
    alpha_bounds: tuple[float, float] = (0.0, np.radians(40.0))

    h_ref: float = 40000.0
    v_ref: float = 2000.0
    theta_defect_ref: float = np.radians(0.5)
    phi_defect_ref: float = np.radians(0.5)
    gamma_defect_ref: float = np.radians(1.0)
    psi_defect_ref: float = np.radians(1.0)

    objective: str = "maximize_final_velocity"  # current behavior
    objective_ref: float = -0.1

    print_driver_results: bool = False
    check_setup: bool = True
    simulate: bool = True


@dataclass
class VehicleRunSpec:
    """
    A single Dymos solve request.

    `initial_state_override` is used for replanning. It should contain keys:
    h, theta, phi, v, gamma, psi, all in SI/radians.
    """
    label: str
    initial: VehicleInitialCondition
    target: TargetCondition
    terminal_guess: TerminalGuess = field(default_factory=TerminalGuess)
    duration_guess: float = 250.0
    fixed_duration: bool = False
    initial_state_override: Optional[Dict[str, float]] = None


def _make_problem(solver_config: DymosSolverConfig) -> tuple[om.Problem, dm.Trajectory, dm.Phase, str]:
    """Build the OpenMDAO/Dymos problem and return problem, trajectory, phase, optimizer."""
    p = om.Problem(model=om.Group())
    _, optimizer = set_pyoptsparse_opt(solver_config.optimizer, fallback=solver_config.fallback_optimizer)

    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = optimizer
    p.driver.options['debug_print'] = []
    p.driver.options['print_results'] = solver_config.print_driver_results

    p.model.linear_solver = om.DirectSolver(rhs_checking=True)
    p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    p.model.nonlinear_solver.options['iprint'] = 0

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
        transcription = dm.Radau(num_segments=solver_config.num_segments_ipopt)
    else:
        transcription = dm.Radau(num_segments=solver_config.num_segments_other,
                                 order=solver_config.order_other)

    phase0 = traj.add_phase('phase0', dm.Phase(ode_class=VehicleODE, transcription=transcription))
    return p, traj, phase0, optimizer


def _configure_phase(phase0: dm.Phase, run: VehicleRunSpec, cfg: DymosSolverConfig) -> None:
    """Apply time, state, control, and objective declarations to a phase."""
    time_options = dict(
        fix_initial=True,
        fix_duration=run.fixed_duration,
        units="s",
        duration_ref=cfg.duration_ref,
    )
    if not run.fixed_duration:
        time_options["duration_bounds"] = cfg.duration_bounds
    phase0.set_time_options(**time_options)

    phase0.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='hdot',
                     lower=0.0, ref=cfg.h_ref, defect_ref=cfg.h_ref)
    phase0.add_state('theta', fix_initial=True, fix_final=True, units='rad', rate_source='thetadot',
                     lower=cfg.theta_bounds[0], upper=cfg.theta_bounds[1], defect_ref=cfg.theta_defect_ref)
    phase0.add_state('phi', fix_initial=True, fix_final=True, units='rad', rate_source='phidot',
                     lower=cfg.phi_bounds[0], upper=cfg.phi_bounds[1], defect_ref=cfg.phi_defect_ref)
    phase0.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
                     lower=0.0, ref=cfg.v_ref, defect_ref=cfg.v_ref)
    phase0.add_state('gamma', fix_initial=True, fix_final=False, units='rad', rate_source='gammadot',
                     lower=cfg.gamma_bounds[0], upper=cfg.gamma_bounds[1], defect_ref=cfg.gamma_defect_ref)
    phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad', rate_source='psidot',
                     lower=cfg.psi_bounds[0], upper=cfg.psi_bounds[1], defect_ref=cfg.psi_defect_ref)

    phase0.add_control('sigma', units='rad', opt=True,
                       lower=cfg.sigma_bounds[0], upper=cfg.sigma_bounds[1], rate_continuity=True)
    phase0.add_control('alpha', units='rad', opt=True,
                       lower=cfg.alpha_bounds[0], upper=cfg.alpha_bounds[1], rate_continuity=True)

    if cfg.objective == "maximize_final_velocity":
        phase0.add_objective('v', loc='final', ref=cfg.objective_ref)
    elif cfg.objective == "minimize_time":
        phase0.add_objective('time', loc='final', ref=cfg.duration_ref)
    else:
        raise ValueError(f"Unknown objective: {cfg.objective}")


def _set_initial_values(phase0: dm.Phase, run: VehicleRunSpec) -> None:
    """Set state/control initial guesses and boundary values."""
    phase0.set_time_val(initial=0.0, duration=run.duration_guess, units='s')

    if run.initial_state_override is None:
        h0 = run.initial.h
        theta0 = run.initial.theta
        phi0 = run.initial.phi
        v0 = run.initial.v
        gamma0 = run.initial.gamma
        psi0 = run.initial.psi
    else:
        h0 = float(run.initial_state_override['h'])
        theta0 = float(run.initial_state_override['theta'])
        phi0 = float(run.initial_state_override['phi'])
        v0 = float(run.initial_state_override['v'])
        gamma0 = float(run.initial_state_override['gamma'])
        psi0 = float(run.initial_state_override['psi'])

    phase0.set_state_val('h', [h0, run.target.h], units='m')
    phase0.set_state_val('theta', [theta0, run.target.theta], units='rad')
    phase0.set_state_val('phi', [phi0, run.target.phi], units='rad')
    phase0.set_state_val('v', [v0, run.terminal_guess.v], units='m/s')
    phase0.set_state_val('gamma', [gamma0, run.terminal_guess.gamma], units='rad')
    phase0.set_state_val('psi', [psi0, run.terminal_guess.psi], units='rad')

    phase0.set_control_val('sigma', [0.0, 0.0], units='rad')
    phase0.set_control_val('alpha', [0.0, 0.0], units='rad')


def solve_vehicle(run: VehicleRunSpec, solver_config: Optional[DymosSolverConfig] = None) -> Dict[str, Any]:
    """
    Build, solve, and simulate one vehicle trajectory.

    This is the importable replacement for the old hard-coded solve_vehicle(...)
    signature. Mission parameters are provided through VehicleRunSpec.
    """
    cfg = solver_config or DymosSolverConfig()
    t0 = time.time()

    p, traj, phase0, optimizer = _make_problem(cfg)
    _configure_phase(phase0, run, cfg)

    p.setup(check=cfg.check_setup)
    _set_initial_values(phase0, run)

    dm.run_problem(p, simulate=cfg.simulate)
    elapsed = time.time() - t0

    print(f"[{run.label}] Optimization time: {elapsed:.2f} s")

    sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
    sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

    t_sim = sim.get_val('traj.phase0.timeseries.time').ravel()
    theta_rad = sim.get_val('traj.phase0.timeseries.theta').ravel()
    phi_rad = sim.get_val('traj.phase0.timeseries.phi').ravel()
    h_sim = sim.get_val('traj.phase0.timeseries.h').ravel()
    v_sim = sim.get_val('traj.phase0.timeseries.v').ravel()
    gamma_rad = sim.get_val('traj.phase0.timeseries.gamma').ravel()
    psi_rad = sim.get_val('traj.phase0.timeseries.psi').ravel()

    out = {
        'label': run.label,
        'time': t_sim,
        'theta': np.degrees(theta_rad),      # retained for current plotting code
        'phi': np.degrees(phi_rad),
        'theta_rad': theta_rad,
        'phi_rad': phi_rad,
        'h': h_sim,
        'v': v_sim,
        'gamma': gamma_rad,
        'psi': psi_rad,
        'gamma_deg': np.degrees(gamma_rad),
        'psi_deg': np.degrees(psi_rad),
        't_final': float(t_sim[-1]),
        'sol': sol,
        'sim': sim,
        'problem': p,
        'trajectory': traj,
        'optimizer': optimizer,
    }

    print(f"[{run.label}] Final time to reach target: {out['t_final']:.3f} s")
    return out


def get_state_at_time(case_or_sim, t_query: float, prefix: str = 'traj.phase0.timeseries.') -> Dict[str, float]:
    """
    Interpolate vehicle state at time t_query.

    Accepts either a result dict returned by solve_vehicle or a Dymos simulation Case.
    Returned angular values are in radians.
    """
    sim_case = case_or_sim['sim'] if isinstance(case_or_sim, dict) else case_or_sim
    t = sim_case.get_val(prefix + 'time').ravel()

    def interp(var: str) -> float:
        y = sim_case.get_val(prefix + var).ravel()
        return float(np.interp(t_query, t, y))

    return {
        'h': interp('h'),
        'theta': interp('theta'),
        'phi': interp('phi'),
        'v': interp('v'),
        'gamma': interp('gamma'),
        'psi': interp('psi'),
    }


def concat_timeseries(case_stage1: Dict[str, Any], case_stage2: Dict[str, Any],
                      var: str, t_cut: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a timeseries variable from two Dymos result dictionaries."""
    sim1 = case_stage1['sim']
    sim2 = case_stage2['sim']
    t1 = sim1.get_val('traj.phase0.timeseries.time').ravel()
    y1 = sim1.get_val(f'traj.phase0.timeseries.{var}').ravel()
    m1 = t1 <= t_cut

    t2 = sim2.get_val('traj.phase0.timeseries.time').ravel() + offset
    y2 = sim2.get_val(f'traj.phase0.timeseries.{var}').ravel()

    return np.concatenate([t1[m1], t2]), np.concatenate([y1[m1], y2])


def stitch_two_stage(case1: Dict[str, Any], case2: Dict[str, Any], label: str,
                     t_reveal: float) -> Dict[str, Any]:
    """Stitch stage-1 up to t_reveal and all of stage-2 shifted by t_reveal."""
    t1 = case1['time']
    m1 = t1 <= t_reveal
    out = {
        'label': label,
        'time': np.concatenate([case1['time'][m1], case2['time'] + t_reveal]),
        'theta': np.concatenate([case1['theta'][m1], case2['theta']]),
        'phi': np.concatenate([case1['phi'][m1], case2['phi']]),
        'theta_rad': np.concatenate([case1['theta_rad'][m1], case2['theta_rad']]),
        'phi_rad': np.concatenate([case1['phi_rad'][m1], case2['phi_rad']]),
        'h': np.concatenate([case1['h'][m1], case2['h']]),
        'v': np.concatenate([case1['v'][m1], case2['v']]),
        'gamma': np.concatenate([case1['gamma'][m1], case2['gamma']]),
        'psi': np.concatenate([case1['psi'][m1], case2['psi']]),
        't_final': t_reveal + case2['t_final'],
        'stage1': case1,
        'stage2': case2,
        'sim1': case1['sim'],
        'sim2': case2['sim'],
    }
    return out


@dataclass
class TwoStageSalvoConfig:
    """Mission-level parameters for the current two-stage Dymos workflow."""
    t_reveal: float = 50.0
    target_theta_guess: float = np.radians(2.01)
    target_phi_guess: float = np.radians(1.05)
    target_theta_true: float = np.radians(2.0)
    target_phi_true: float = np.radians(1.0)
    terminal_gamma_guess: float = -np.radians(45.0)
    terminal_psi_guess: float = 0.0

    vehicle_A_initial: VehicleInitialCondition = field(default_factory=lambda: VehicleInitialCondition(
        phi=np.radians(1.7), psi=0.0))
    vehicle_B_initial: VehicleInitialCondition = field(default_factory=lambda: VehicleInitialCondition(
        phi=0.0 / 6_371_000.0, psi=0.0))


def make_stage_run(label: str, initial: VehicleInitialCondition, target_theta: float, target_phi: float,
                   terminal_gamma: float, terminal_psi: float,
                   duration_guess: float = 250.0,
                   fixed_duration: bool = False,
                   initial_state_override: Optional[Dict[str, float]] = None) -> VehicleRunSpec:
    return VehicleRunSpec(
        label=label,
        initial=initial,
        target=TargetCondition(theta=target_theta, phi=target_phi, h=0.0),
        terminal_guess=TerminalGuess(v=800.0, gamma=terminal_gamma, psi=terminal_psi),
        duration_guess=duration_guess,
        fixed_duration=fixed_duration,
        initial_state_override=initial_state_override,
    )


def run_two_stage_salvo(mission: Optional[TwoStageSalvoConfig] = None,
                        solver_config: Optional[DymosSolverConfig] = None,
                        synchronize_salvo: bool = True) -> Dict[str, Any]:
    """
    Run the two-stage, two-vehicle Dymos workflow.

    If synchronize_salvo=True, the slower stage-1 vehicle defines the salvo time,
    and the quicker stage-1 vehicle is re-solved with fixed duration equal to that time.
    """
    mission = mission or TwoStageSalvoConfig()
    solver_config = solver_config or DymosSolverConfig()

    A1 = solve_vehicle(make_stage_run(
        "Vehicle A (stage 1)", mission.vehicle_A_initial,
        mission.target_theta_guess, mission.target_phi_guess,
        mission.terminal_gamma_guess, mission.terminal_psi_guess), solver_config)

    B1 = solve_vehicle(make_stage_run(
        "Vehicle B (stage 1)", mission.vehicle_B_initial,
        mission.target_theta_guess, mission.target_phi_guess,
        mission.terminal_gamma_guess, mission.terminal_psi_guess), solver_config)

    salvo_time = max(A1['t_final'], B1['t_final'])

    if synchronize_salvo:
        if A1['t_final'] < salvo_time:
            A1 = solve_vehicle(make_stage_run(
                "Vehicle A (stage 1 synchronized)", mission.vehicle_A_initial,
                mission.target_theta_guess, mission.target_phi_guess,
                mission.terminal_gamma_guess, mission.terminal_psi_guess,
                duration_guess=salvo_time, fixed_duration=True), solver_config)
        if B1['t_final'] < salvo_time:
            B1 = solve_vehicle(make_stage_run(
                "Vehicle B (stage 1 synchronized)", mission.vehicle_B_initial,
                mission.target_theta_guess, mission.target_phi_guess,
                mission.terminal_gamma_guess, mission.terminal_psi_guess,
                duration_guess=salvo_time, fixed_duration=True), solver_config)

    A_reveal = get_state_at_time(A1, mission.t_reveal)
    B_reveal = get_state_at_time(B1, mission.t_reveal)

    A2_duration_guess = max(solver_config.duration_bounds[0], A1['t_final'] - mission.t_reveal)
    B2_duration_guess = max(solver_config.duration_bounds[0], B1['t_final'] - mission.t_reveal)

    A2 = solve_vehicle(make_stage_run(
        "Vehicle A (stage 2)", mission.vehicle_A_initial,
        mission.target_theta_true, mission.target_phi_true,
        mission.terminal_gamma_guess, mission.terminal_psi_guess,
        duration_guess=A2_duration_guess,
        initial_state_override=A_reveal), solver_config)

    B2 = solve_vehicle(make_stage_run(
        "Vehicle B (stage 2)", mission.vehicle_B_initial,
        mission.target_theta_true, mission.target_phi_true,
        mission.terminal_gamma_guess, mission.terminal_psi_guess,
        duration_guess=B2_duration_guess,
        initial_state_override=B_reveal), solver_config)

    A = stitch_two_stage(A1, A2, "Vehicle A (two-stage)", mission.t_reveal)
    B = stitch_two_stage(B1, B2, "Vehicle B (two-stage)", mission.t_reveal)

    return {
        'mission': mission,
        'solver_config': solver_config,
        'A1': A1,
        'B1': B1,
        'A2': A2,
        'B2': B2,
        'A': A,
        'B': B,
        'salvo_time': salvo_time,
    }


# This module intentionally does not run any mission on import.
# Put mission execution in main.py or in tests.
