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

class LoSDistance(om.ExplicitComponent):
    """
    For each node i in Traj B, compute the minimum 3D distance between the
    line-of-sight line from P_B(i) to P_T (target) and nearby sampled
    points Q_j of Traj A.

    Inputs:
      theta  [rad], phi [rad], h [m]   (B trajectory states)

    Output:
      los_min_dist [m]  (minimum distance from LoS(B->Target) to Traj A)

    Options:
      earth_radius : float (m)
      theta_A : np.ndarray [rad]
      phi_A   : np.ndarray [rad]
      h_A     : np.ndarray [m]
      target_theta : float [rad]
      target_phi   : float [rad]
      target_h     : float [m]
      window : int  (half-width in A indices to search around mapped index)
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('earth_radius', default=6_371_000.0)
        self.options.declare('theta_A', types=np.ndarray)
        self.options.declare('phi_A',   types=np.ndarray)
        self.options.declare('h_A',     types=np.ndarray)
        self.options.declare('target_theta', types=float)
        self.options.declare('target_phi',   types=float)
        self.options.declare('target_h',     types=float)
        self.options.declare('window', default=10, types=int)   # +/- 10 indices

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('theta', val=np.zeros(nn), units='rad')
        self.add_input('phi',   val=np.zeros(nn), units='rad')
        self.add_input('h',     val=np.zeros(nn), units='m')

        self.add_output('los_min_dist', val=np.zeros(nn), units='m')

        ar = np.arange(nn, dtype=int)
        # FD is fine; change to 'cs' if your model supports complex step
        self.declare_partials('los_min_dist', 'theta', rows=ar, cols=ar, method='fd')
        self.declare_partials('los_min_dist', 'phi',   rows=ar, cols=ar, method='fd')
        self.declare_partials('los_min_dist', 'h',     rows=ar, cols=ar, method='fd')

        # Precompute Traj A Cartesian points
        Re = self.options['earth_radius']
        theta_A = self.options['theta_A'].reshape(-1)
        phi_A   = self.options['phi_A'].reshape(-1)
        h_A     = self.options['h_A'].reshape(-1)

        rA   = Re + h_A
        cphi = np.cos(phi_A); sphi = np.sin(phi_A)
        cth  = np.cos(theta_A); sth = np.sin(theta_A)
        self._QA = np.vstack((rA*cphi*cth, rA*cphi*sth, rA*sphi)).T  # (N_A, 3)
        self._NA = self._QA.shape[0]

        # Index mapping factor if N_A != N_B (maps B index -> A index)
        self._NB = nn
        if nn > 1:
            self._index_scale = (self._NA - 1) / (nn - 1)
        else:
            self._index_scale = 0.0

        # Target Cartesian
        thT = self.options['target_theta']
        phT = self.options['target_phi']
        hT  = self.options['target_h']
        rT  = Re + hT
        self._PT = np.array([rT*np.cos(phT)*np.cos(thT),
                             rT*np.cos(phT)*np.sin(thT),
                             rT*np.sin(phT)])


    def compute(self, inputs, outputs):
        theta = inputs['theta']
        phi   = inputs['phi']
        h     = inputs['h']
    
        nn = len(theta)
        los_dist = outputs['los_min_dist']
        QA = self._QA
        PT = self._PT
    
        halfwin = 2  # your window half-width
    
        for i in range(nn):
    
            # Skip first and last 20 nodes
            if i < 0 or i >= nn - 20:
                los_dist[i] = 5e1  # or even np.nan, but IPOPT dislikes NaNs
                continue
    
            # Convert B node to Cartesian
            Re = self.options['earth_radius']
            rB = Re + h[i]
            cphi = np.cos(phi[i])
            sphi = np.sin(phi[i])
            cth = np.cos(theta[i])
            sth = np.sin(theta[i])
            PB = np.array([rB*cphi*cth, rB*cphi*sth, rB*sphi])
    
            # Direction of line to target
            d = PT - PB
            L = np.linalg.norm(d)
            if L < 1e-9:
                los_dist[i] = 0.0
                continue
            d = d / L
    
            # Window of A indices
            j1 = max(0, i - halfwin)
            j2 = min(QA.shape[0], i + halfwin + 1)
            QA_sub = QA[j1:j2]
    
            # Vector differences
            v = QA_sub - PB
            t = np.clip(np.einsum('ij,j->i', v, d), 0.0, L)
            proj = PB + np.outer(t, d)
            diff = QA_sub - proj
            dist2 = np.sum(diff*diff, axis=1)
    
            los_dist[i] = np.sqrt(np.min(dist2))


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
        self.options.declare('los_opts', default=None)

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

        # Optional LoS distance from B->Target vs Traj A cloud
        los_opts = self.options['los_opts']
        if los_opts is not None:
            self.add_subsystem(
                'los_dist',
                LoSDistance(num_nodes=nn, **los_opts),
                promotes_inputs=['theta', 'phi', 'h'],      # from B states
                promotes_outputs=['los_min_dist']           # scalar per node
            )


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


def solve_vehicle(initial_phi, final_theta, final_phi, final_gamma, final_psi, 
                  t_duration=2000.0, label="Case", los_opts=None):
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
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    p.model.nonlinear_solver.options['iprint'] = 0

    # (Optional) IPOPT options
    if optimizer == 'IPOPT':
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['linear_solver'] = 'mumps'
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['mu_strategy'] = 'adaptive'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['mu_init'] = 0.1
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['constr_viol_tol'] = 1e-4
        p.driver.opt_settings['compl_inf_tol'] = 1e-4
        p.driver.opt_settings['tol'] = 1e-5

    traj = p.model.add_subsystem('traj', dm.Trajectory())
    if optimizer == 'IPOPT':
        phase0 = traj.add_phase(
                                'phase0',
                                dm.Phase(
                                    ode_class=VehicleODE,
                                    ode_init_kwargs={'los_opts': los_opts},  
                                    transcription=dm.Radau(num_segments=10)
                                )
                            )
    else:
        phase0 = traj.add_phase(
                                'phase0',
                                dm.Phase(
                                    ode_class=VehicleODE,
                                    ode_init_kwargs={'los_opts': los_opts},   
                                    transcription=dm.Radau(num_segments=15, order=3)
                                )
                            )

    # Time/state/control defs
    phase0.set_time_options(fix_initial=True, fix_duration=False, units='s',
                        duration_ref=500.0, duration_bounds=(250.0, 1000.0))

    phase0.add_state('h',     fix_initial=True, fix_final=True,  units='m',  rate_source='hdot',     lower=0, ref=40000, defect_ref=4.0e4)
    phase0.add_state('theta', fix_initial=True, fix_final=True,  units='rad', rate_source='thetadot', lower=0, upper=np.radians(6), defect_ref=np.radians(0.5))
    phase0.add_state('phi',   fix_initial=True, fix_final=True,  units='rad', rate_source='phidot',   lower=-np.radians(0), upper=np.radians(6), defect_ref=np.radians(0.5))
    phase0.add_state('v',     fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',     lower=0, ref=2000, defect_ref=2000)
    phase0.add_state('gamma', fix_initial=True, fix_final=False,  units='rad', rate_source='gammadot', lower=-np.radians(89), upper=np.radians(89), defect_ref=np.radians(5))
    phase0.add_state('psi',   fix_initial=True, fix_final=False,  units='rad', rate_source='psidot',   lower=-np.radians(0), upper=np.radians(90), defect_ref=np.radians(10))
    phase0.add_control('sigma', units='rad', opt=True, lower=np.radians(-1), upper=np.radians(165), rate_continuity=True)
    phase0.add_control('alpha', units='rad', opt=True, lower=np.radians(0), upper=np.radians(40), rate_continuity=True)

    if los_opts is not None:
        phase0.add_timeseries_output('los_min_dist', shape=(1,))
        phase0.add_path_constraint(
        'los_min_dist',
        lower=20.0,
        units='m',
        ref=50.0,        # scaling for the constraint (tune as needed)
    )


    # Objective Function
    phase0.add_objective('time', loc='final', ref=0.1)

    p.setup(check=True)

    # Initial and terminal boundary values (share initial, vary final)
    phase0.set_time_val(initial=0.0, duration=t_duration, units='s')
    # Initials (same for both vehicles)
    phase0.set_state_val('h',     [40000.0, 0.0],           units='m')
    phase0.set_state_val('theta', [0.0,     final_theta],   units='rad')
    phase0.set_state_val('phi',   [initial_phi,     final_phi],     units='rad')
    phase0.set_state_val('v',     [2000.0,  800.0],        units='m/s')   # final free but initialized
    phase0.set_state_val('gamma', [0.0,     final_gamma],   units='rad')
    phase0.set_state_val('psi',   [np.radians(0), final_psi], units='rad')

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

    return {
        'label': label,
        'theta': theta_sim,
        'phi':   phi_sim,
        'h':     h_sim,
        'time':  t_sim,
        'sol':   sol,
        'sim':   sim,
    }

def ts(case, var):
    """Return (t, y) from the simulation timeseries for a var name like 'alpha' or 'sigma'."""
    sim = case['sim']
    t = sim.get_val('traj.phase0.timeseries.time').ravel()
    y = sim.get_val(f'traj.phase0.timeseries.{var}').ravel()
    return t, y

# Define two terminal-condition variants
case_A = solve_vehicle(initial_phi = np.radians(0.0),
                        final_theta=np.radians(1.5),
                       final_phi=np.radians(2.0),
                       final_gamma=-np.radians(45),
                       final_psi=np.radians(85),
                       label="Vehicle A")

# Record the A trajectory
A_sim = case_A['sim']
theta_A = A_sim.get_val('traj.phase0.timeseries.theta').ravel()
phi_A   = A_sim.get_val('traj.phase0.timeseries.phi').ravel()
h_A     = A_sim.get_val('traj.phase0.timeseries.h').ravel()

los_opts = dict(
    earth_radius=6_371_000.0,
    theta_A=theta_A,
    phi_A=phi_A,
    h_A=h_A,
    target_theta=np.radians(3.5),
    target_phi=np.radians(2.1),
    target_h=0.0,
    window=1
)

earth_radius=6_371_000.0
case_B = solve_vehicle(initial_phi = 30.0/earth_radius,
                        final_theta=np.radians(1.5),
                       final_phi=np.radians(2.0),
                       final_gamma=-np.radians(45),
                       final_psi=np.radians(85),
                       label="Vehicle B",
                       los_opts=los_opts)

B_sim = case_B['sim']
tB    = B_sim.get_val('traj.phase0.timeseries.time').ravel()
dLoS  = B_sim.get_val('traj.phase0.timeseries.los_min_dist').ravel()

tA, alphaA = ts(case_A, 'alpha')
tB, alphaB = ts(case_B, 'alpha')

tA_s, sigmaA = ts(case_A, 'sigma')
tB_s, sigmaB = ts(case_B, 'sigma')

# Convert to degrees if you prefer
alphaA_deg = np.degrees(alphaA)
alphaB_deg = np.degrees(alphaB)
sigmaA_deg = np.degrees(sigmaA)
sigmaB_deg = np.degrees(sigmaB)

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

# Plot LoS Constraint Value vs. Time
plt.figure(figsize=(8,5))
plt.plot(tB, dLoS, lw=2)
plt.xlabel('Time along Traj B (s)')
plt.ylabel('Min distance (m) from LoS(B -> Target) to Traj A')
plt.title('LoS Clearance vs Time (Traj B)')
plt.grid(True); plt.tight_layout()

# Plot Top Down View
plt.figure(figsize=(8,6))
plt.plot(case_A['theta'], case_A['phi'], '-',  label=case_A['label'])
plt.plot(case_B['theta'], case_B['phi'], '-',  label=case_B['label'])
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
