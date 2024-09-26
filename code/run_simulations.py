import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, cholesky

class GenerateDGP():
    def __init__(self, n: int, T:int, alpha: float, num_controls: int = None, num_instruments: int = None):
        self.n = n
        self.T = T
        self.num_controls = num_controls
        self.num_instruments = num_instruments
        self.alpha = alpha

    def generate_individual_heterogeneity(self, n, T):
        mean = np.zeros(n)
        cov = np.array([[0.5**abs(i-j) for j in range(n)] for i in range(n)]) * (4/T)
        e = np.random.multivariate_normal(mean, cov)
        return e

    def generate_disturbances(self, n, T, rho_e, rho_u, rho_nu=0.5):
        # Initialize epsilon and u
        epsilon = np.zeros((n, T))
        u = np.zeros((n, T))
        
        # Covariance matrix for (nu1, nu2)
        cov_matrix = np.array([[1, rho_nu], [rho_nu, 1]])
        
        # Generate initial conditions
        nu = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=(n, T))
        
        for i in range(n):
            for t in range(T):
                if t == 0:
                    # Assuming initial conditions are drawn from stationary distribution
                    epsilon[i, t] = nu[i, t, 0] / np.sqrt(1 - rho_e**2)
                    u[i, t] = nu[i, t, 1] / np.sqrt(1 - rho_u**2)
                else:
                    epsilon[i, t] = rho_e * epsilon[i, t-1] + nu[i, t, 0]
                    u[i, t] = rho_u * u[i, t-1] + nu[i, t, 1]
                    
        return epsilon, u
    
    def generate_controls(self, n, T, px, rho_x, e):
        x = np.zeros((n, T, px))
        
        for i in range(n):
            for j in range(px):
                phi = np.random.normal(size=T)
                for t in range(T):
                    if t == 0:
                        x[i, t, j] = e[i] / (1 - rho_x) + np.sqrt(1 / (1 - rho_x**2)) * phi[t]
                    else:
                        x[i, t, j] = e[i] + rho_x * x[i, t-1, j] + phi[t]
                        
        return x
    
    def generate_instruments(self, n, T, x, px, pz):
        # Generate z variables as z_{it} = Π x_{it} + ζ_{it}
        Pi = np.hstack([np.eye(pz), np.zeros((pz, px - pz))])
        zeta = np.random.multivariate_normal(mean=np.zeros(pz), cov=0.25 * np.eye(pz), size=(n, T))
        z = np.zeros((n, T, pz))
        for t in range(T):
            z[:, t, :] = np.dot(x[:, t, :], Pi.T) + zeta[:, t, :]

        return z

    def generate_post_selection_regularization_dgp(self, n=100, T=10, px=50, pz=5, rho_e=0.8, rho_u=0.8, rho_x=0.8):
        # Initialize lists to store results
        results = []
        
        e = self.generate_individual_heterogeneity(n, T)
        epsilon, u = self.generate_disturbances(n, T, rho_e, rho_u)
        x = self.generate_controls(n, T, px, rho_x, e)
        z = self.generate_instruments(n, T, x, px, pz)
        
        # Generate d and y
        gamma = (1.0 / (np.arange(1, px + 1))) ** 2
        delta = (1.0 / (np.arange(1, pz + 1))) ** 2
        alpha = self.alpha
        beta = gamma
        
        d = np.zeros((n, T))
        y = np.zeros((n, T))
        
        for t in range(T):
            d[:, t] = e + z[:, t, :].dot(delta) + x[:, t, :].dot(gamma) + u[:, t]
            y[:, t] = e + alpha * d[:, t] + x[:, t, :].dot(beta) + epsilon[:, t]
        
        # Store data in results list
        for t in range(T):
            df_t = pd.DataFrame({
                'y': y[:, t],
                'd': d[:, t],
                't': t + 1,
                'unit': np.arange(1, n + 1)
            })
            df_t = pd.concat([df_t, pd.DataFrame(z[:, t, :], columns=[f'instrument_{j+1}' for j in range(pz)])], axis=1)
            df_t = pd.concat([df_t, pd.DataFrame(x[:, t, :], columns=[f'x_{j+1}' for j in range(px)])], axis=1)
            results.append(df_t)
        
        # Combine all results into a single DataFrame
        df = pd.concat(results, ignore_index=True)
        
        return df

    def generate_data_chernozhukov(self, num_instruments:int = None) -> pd.DataFrame:
        """Chernozhukov High-Dimensional Panel Data DGP

        Args:
            n (int): Number of units/individuals
            T (int): Length of panel
            alpha (float): Coefficient to be found
            num_instruments (int, optional): Number of instruments to be set. Defaults to None.

        Returns:
            pd.DataFrame: Simulated data
        """
        
        # Parameters
        rho_epsilon = 0.8
        rho_u = 0.8
        rho_z = 0.8
        rho_v = 0.5
        alpha = self.alpha

        if num_instruments:
            num_instruments = num_instruments
        else:
            num_instruments= self.n * (self.T - 2)

        print(f"Num instruments: {num_instruments}")

        s = np.ceil(0.5 * self.n**(1/3)).astype(int)
        
        # Generate disturbances
        epsilon = np.zeros((self.n, self.T))
        u = np.zeros((self.n, self.T))
        for i in range(self.n):
            for t in range(1, self.T):
                cov_matrix = np.array([[1, rho_v], [rho_v, 1]])
                residuals = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix)
                epsilon[i, t] = rho_epsilon * epsilon[i, t-1] + residuals[0]
                u[i, t] = rho_u * u[i, t-1] + residuals[1]
        
        # Generate individual heterogeneity (fixed effects)
        mean_e = np.zeros(self.n)
        cov_e = 0.5 ** np.abs(np.subtract.outer(np.arange(self.n), np.arange(self.n)))
        e = np.random.multivariate_normal(mean=mean_e, cov=cov_e) * (4 / self.T)**0.5
        
        # Design of coefficients on the instruments
        pi1 = np.zeros(num_instruments)
        for j in range(1, num_instruments + 1):
            if j <= s:
                pi1[j-1] = (-1)**(j-1) * (1 / np.sqrt(s))
            else:
                pi1[j-1] = (-1)**(j-1) * (1 / j**2)
        
        # Generate instruments
        z = np.zeros((self.n, self.T, num_instruments))
        for i in range(self.n):
            for t in range(self.T):
                phi_t = np.random.normal(scale=np.sqrt(1), size=num_instruments)
                for j in range(num_instruments):
                    if t == 0:
                        z[i, t, j] = e[i] / (1 - rho_z) + np.sqrt(1 / (1 - rho_z**2)) * phi_t[j]
                    else:
                        z[i, t, j] = e[i] + rho_z * z[i, t-1, j] + phi_t[j]
        
        # Generate endogenous variable
        d = np.sum(z * pi1[:num_instruments], axis=2) + u + e[:, np.newaxis]
        
        # Generate dependent variable
        y = alpha * d + e[:, np.newaxis] + epsilon

        # Prepare DataFrame
        df = pd.DataFrame({
            'Individual': np.repeat(np.arange(self.n), self.T),
            'Time': np.tile(np.arange(self.T), self.n),
            'Y': y.flatten(),
            'D': d.flatten(),
        })

        # Create a dictionary to hold all the instrument columns
        instrument_cols = {f'instrument_{j+1}': z[:, :, j].flatten() for j in range(num_instruments)}

        # Convert the dictionary to a DataFrame
        instruments_df = pd.DataFrame(instrument_cols)

        # Use pd.concat to add these new columns to the existing DataFrame
        df = pd.concat([df, instruments_df], axis=1)
        
        return df