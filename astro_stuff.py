import numpy as np
import matplotlib.pyplot as plt
from integration import Integrator

J2 = 1082.6e-06
EARTH_RADIUS = 6378.137  # km (WGS84 equatorial radius)
EARTH_MU = 3.986e05  # km^3/s^2


class Satellite:
    """Satellite with Keplerian orbital elements."""
    
    def __init__(self, name, a, e, i, raan, omega, M0, n=None, epoch=0):
        self.name = name
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.omega = omega
        self.M0 = M0
        self.n = n if n is not None else np.sqrt(EARTH_MU / a**3)
        self.epoch = epoch

    def get_state(self):
        """Return current orbital elements as array"""
        return np.array([self.a, self.e, self.i, self.raan, self.omega, self.M0])
    
    def set_state(self, state):
        """Set orbital elements from array"""
        self.a, self.e, self.i, self.raan, self.omega, self.M0 = state


    @staticmethod
    def first_order_secular_rates(t, state_flat) -> np.array:
        """
        Calculate first-order secular rates due to J2
        
        Parameters:
        -----------
        t : float (time, not used for secular rates but required by integrator)
        state_flat : np.array of shape (n_satellites * 6,)
                    Flattened array: [a1, e1, i1, raan1, omega1, M01, 
                                    a2, e2, i2, raan2, omega2, M02, ...]
        
        Returns:
        --------
        rates_flat : np.array of shape (n_satellites * 6,)
        """
        # Reshape to (n_satellites, 6)
        n_sats = len(state_flat) // 6
        states = state_flat.reshape(n_sats, 6)
        
        rates = np.zeros_like(states)
        
        a = states[:, 0]
        e = states[:, 1]
        i = states[:, 2]
        
        n = np.sqrt(EARTH_MU / a**3)
        
        # Common terms
        cos_i = np.cos(i)
        cos_i_sq = cos_i**2
        ecc_term = (1 - e**2)**2
        radius_ratio_sq = (EARTH_RADIUS / a)**2
        n_J2 = n * J2 * radius_ratio_sq
        
        # No secular rates for a, e, i
        rates[:, 0] = 0  # da/dt
        rates[:, 1] = 0  # de/dt
        rates[:, 2] = 0  # di/dt
        
        # RAAN rate
        rates[:, 3] = -1.5 * n_J2 * cos_i / ecc_term
        
        # Argument of perigee rate
        rates[:, 4] = -0.75 * n_J2 * (1 - 5*cos_i_sq) / ecc_term
        
        # Mean anomaly rate
        rates[:, 5] = n + 0.75 * n_J2 * (3*cos_i_sq - 1) / (4 * (1 - e**2)**1.5)
        
        # Flatten back to 1D array
        return rates.flatten()


class GroundStation:
    """Ground station with geodetic coordinates and visibility constraints."""
    
    def __init__(self, name, lat_deg, lon_deg, alt):
        self.name = name
        self.lat = np.radians(lat_deg)
        self.lon = np.radians(lon_deg)
        self.alt = alt
    

# Example usage:
if __name__ == "__main__":
    # Create 20+ satellites with different orbits
    satellites = []
    
    # Generate a constellation of satellites
    for sat_num in range(1):
        # Create satellites with varying parameters
        a = 12000  # Semi-major axis between 7000-8000 km
        e = 0.3  # Low eccentricity
        i = np.radians(20)  # Inclination around 20 deg
        raan = np.radians(40)  # Random RAAN
        omega = np.radians(60)  # Random argument of perigee
        M0 = np.radians(80)  # Random mean anomaly
        
        sat = Satellite(
            name=f"Sat{sat_num:02d}",
            a=a, e=e, i=i, raan=raan, omega=omega, M0=M0
        )
        satellites.append(sat)
    
    # Create initial state matrix for all satellites and FLATTEN it
    initial_states_list = [sat.get_state() for sat in satellites]
    initial_states_flat = np.concatenate(initial_states_list)  # This gives shape (n_sats * 6,)
    
    # Propagate all satellites at once using batch processing
    times, states_flat = Integrator.propagate_ode(
        ode=Satellite.first_order_secular_rates,
        state0=initial_states_flat,  # Now a 1D array
        tspan=5e4,
        dt=1,
        method='rk4'
    )
    
    # Reshape the results for easier analysis
    n_sats = len(satellites)
    n_steps = len(times)
    # states_flat has shape (n_steps, n_sats * 6)
    # Reshape to (n_steps, n_sats, 6) for easier indexing
    states = states_flat.reshape(n_steps, n_sats, 6)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot RAAN for all satellites
    for sat_idx in range(n_sats):
        raan = np.degrees(states[:, sat_idx, 3])
        axes[0].plot(times, raan, alpha=0.5, linewidth=0.8, label=f'Sat{sat_idx:02d}')
    
    axes[0].set_ylabel('RAAN (degrees)')
    axes[0].set_title(f'Secular J2 Effects on {n_sats} Satellites - RAAN Drift')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Time (sec)')
    
    # Plot argument of perigee for all satellites
    for sat_idx in range(n_sats):
        omega = np.degrees(states[:, sat_idx, 4])
        axes[1].plot(times, omega, alpha=0.5, linewidth=0.8)
    
    axes[1].set_ylabel('Argument of Perigee (degrees)')
    axes[1].set_title('Argument of Perigee Drift')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.show()
    
    # Create a summary table of RAAN drift rates
    print("\n=== RAAN Drift Rates Summary ===")
    print(f"{'Satellite':<10} {'Initial RAAN (deg)':<20} {'Final RAAN (deg)':<20}")
    print("-" * 65)
    
    for sat_idx, sat in enumerate(satellites):
        initial_raan = np.degrees(states[0, sat_idx, 3])
        final_raan = np.degrees(states[-1, sat_idx, 3])
        drift_per_sec = (final_raan - initial_raan) / 5e4
        
        print(f"{sat.name:<10} {initial_raan} {final_raan} {drift_per_sec}")
    