import numpy as np

class Integrator:
    
    @staticmethod
    def euler_step(f, t, y, dt):
        """
        Simple Euler integration: y(t+dt) = y(t) + dy/dt * dt
        
        Args:
            f: ODE function f(t, y) that returns dy/dt
            t: Current time
            y: Current state
            dt: Time step
            
        Returns:
            Updated state
        """
        return y + f(t, y) * dt
    
    @staticmethod
    def rk4_step(f, t, y, dt):
        """
        Runge-Kutta 4th order integration step
        
        Args:
            f: ODE function f(t, y) that returns dy/dt
            t: Current time
            y: Current state
            dt: Time step
            
        Returns:
            Updated state
        """
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
        k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
        k4 = f(t + dt, y + k3 * dt)
        
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    @staticmethod
    def propagate_ode(ode, state0, tspan, dt, method='rk4'):
        """
        Propagate ODE from initial state over time span.
        
        Args:
            ode: Function f(t, y) that returns state derivatives
            state0: Initial state vector
            tspan: Total time span to integrate
            dt: Time step size
            method: Integration method ('euler' or 'rk4')
            
        Returns:
            times: Array of time points
            states: Array of state vectors at each time point
        """
        # Select integration function
        if method == 'euler':
            step_func = Integrator.euler_step
        elif method == 'rk4':
            step_func = Integrator.rk4_step
        else:
            raise ValueError(f"Unknown method: {method}. Use 'euler' or 'rk4'")
        
        # Setup time array and state storage
        times = np.arange(0, tspan, dt)
        steps = len(times)
        states = np.zeros((steps, len(state0)))
        states[0] = state0
        
        # Integrate
        for step in range(steps - 1):
            states[step + 1] = step_func(ode, times[step], states[step], dt)
        
        return times, states