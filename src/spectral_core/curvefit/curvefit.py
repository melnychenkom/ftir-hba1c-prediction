import numpy as np
import pandas as pd
from jaxfit import CurveFit
import jax.numpy as jnp
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import plotly.graph_objects as go

class SpectraFit:
    def __init__(self):
        self.x_values = None
        self.y_values = None
        self.peaks = None
        self.params = None
        self.pcov = None
        self.predicted = None
        self.discrepancy = None
        self.r2 = None
        self.initial_guess = None
        self.model_loaded = False

    @staticmethod
    def combined_pseudo_voigt(x, *params):
        """ 
        Voigt profile approximation by 
        linear combination of Lorenzian and Gaussian distributions. 
        ------------------------------------------------------------
        Arguments:
            params - voigt profile parameters 
            (mu, gamma, amplitude, eta)
            mu - center of the distribution,
            gamma - FWHM for both gaussian and lorentzian terms,
            amplitude - peak height,
            eta - mixing parameter of the gaussian and lorentzian term

        Returns:
            jnp array
        """
        N = len(params) // 4
        result = jnp.zeros_like(x)
        for i in range(N):
            mu, gamma, amplitude, eta = params[i*4:(i+1)*4]

            a_G = (2 / gamma) * jnp.sqrt(jnp.log(2) / jnp.pi)
            b_G = (4 * jnp.log(2)) / (gamma**2)
            gaussian_term = a_G * jnp.exp(-b_G * (x - mu)**2)

            lorentzian_term = (1 / jnp.pi) * ((gamma / 2) / ((x - mu)**2 + (gamma / 2)**2))

            result += amplitude * (eta * gaussian_term + (1 - eta) * lorentzian_term)

        return result
    
    @staticmethod
    def combined_pseudo_voigt_height_normalized(x, *params):
        """ 
        Pseudo-Voigt profile(s) where 'amplitude' is the peak height.
        Linear combination of height-normalized Lorenzian and Gaussian.
        ------------------------------------------------------------
        Arguments:
            x : jnp array - Evaluation points
            params : tuple - voigt profile parameters flattened
                     (mu_1, gamma_1, amplitude_1, eta_1, mu_2, ...)
                     mu - center of the distribution,
                     gamma - FWHM for both gaussian and lorentzian terms,
                     amplitude - peak height at x = mu,
                     eta - mixing parameter (1=Gaussian, 0=Lorentzian)

        Returns:
            jnp array
        """
        N = len(params) // 4
        result = jnp.zeros_like(x)
        for i in range(N):
            mu, gamma, amplitude, eta = params[i*4:(i+1)*4]

            gaussian_term = jnp.exp(-4.0 * jnp.log(2.0) * ((x - mu) / gamma)**2)
            lorentzian_term = (gamma / 2.0)**2 / ((x - mu)**2 + (gamma / 2.0)**2)

            result += amplitude * (eta * gaussian_term + (1.0 - eta) * lorentzian_term)
        return result

    def _create_params(self, x_values, y_values, peaks, param_dict=None):
        """
        Creates initial parameters and bounds for the fit given a list of peaks.
        ------------------------------------------------------------------------
        Arguments:
            x_values -- wavenumbers
            y_values -- absorbance
            peaks -- peak indices
            param_dict -- dictionary specifying parameter values and bounds
        
        Returns:
            lower & upper bounds and initial values
        """

        default_params = {
            "center": {"min": 5, "max": 5},
            "fwhm": {"min": 0, "max": np.inf, "value": 5},
            "amplitude": {"min": 0, "max": np.inf},
            "eta": {"min": 0, "max": 1, "value": 0.5}
        }

        if param_dict:
            for key, value in param_dict.items():
                if key in default_params:
                    default_params[key].update(value)
                else:
                    print(f"Parameter {key} is not found. Proceeding with default parameters.")

        self.initial_guess = []
        self.lower_bound = []
        self.upper_bound = []

        for peak in peaks:
            wavenumber = x_values[peak]
            amplitude = y_values[peak]

            self.initial_guess.extend([
                wavenumber,                           
                default_params["fwhm"]["value"],       
                amplitude,                                
                default_params["eta"]["value"]           
            ])

            self.lower_bound.extend([
                wavenumber - default_params["center"]["min"], 
                default_params["fwhm"]["min"],              
                default_params["amplitude"]["min"],           
                default_params["eta"]["min"]                   
            ])

            self.upper_bound.extend([
                wavenumber + default_params["center"]["max"],  
                default_params["fwhm"]["max"],                 
                default_params["amplitude"]["max"],             
                default_params["eta"]["max"]                    
            ])

        return None
    
    def load_model(self, model):
        """
        Loads the bounds and initial guess from another model, 
        disregarding any parameters provided in param_dict. 
        This function is intended for fitting spectra based on a previous fit.
        ---------------------------------------------------------------------
        Arguments:
            model - an instance of SpectraFit
        """
        if isinstance(model, SpectraFit):
            try:
                self.initial_guess = model.params_array
                self.lower_bound = model.lower_bound
                self.upper_bound = model.upper_bound
                self.model_loaded = True
            except Exception as e:
                print(f"The following error occured while loading the model: {e}")
        else:
            raise ValueError("model has to be an instance of SpectraFit")
        return None
    

    def load_model_dataframe(self, df, param_dict=None):
        """
        Loads the bounds and initial guess from dataframe of previous fit, 
        disregarding any parameters provided in param_dict. 
        This function is intended for fitting spectra based on a previous fit.
        ---------------------------------------------------------------------
        Arguments:
            df - dataframe of fit of 1 spectra
        """
        default_params = {
            "center": {"min": 5, "max": 5},
            "fwhm": {"min": 0, "max": np.inf, "value": 5},
            "amplitude": {"min": 0, "max": np.inf},
            "eta": {"min": 0, "max": 1, "value": 0.5}
        }

        if param_dict:
            for key, value in param_dict.items():
                if key in default_params:
                    default_params[key].update(value)
                else:
                    print(f"Parameter {key} is not found. Proceeding with default parameters.")
                    
        try:
            fit_initial_guess = []
            fit_lower_bound = []
            fit_upper_bound = []

            for idx, row in df.iterrows():
                wavenumber = row['wavenumber']
                amplitude = row['amplitude']
                fwhm = row['FWHM']
                eta = row['eta']
            
                fit_initial_guess.extend([
                    wavenumber,                               
                    fwhm,                                     
                    amplitude,                                 
                    eta                                       
                ])
                
                fit_lower_bound.extend([
                    wavenumber - default_params["center"]["min"],   
                    default_params["fwhm"]["min"],                  
                    default_params["amplitude"]["min"],             
                    default_params["eta"]["min"]                    
                ])
                
                fit_upper_bound.extend([
                    wavenumber + default_params["center"]["max"],   
                    default_params["fwhm"]["max"],                  
                    default_params["amplitude"]["max"],             
                    default_params["eta"]["max"]                   
                ])
                
            self.initial_guess = fit_initial_guess
            self.lower_bound = fit_lower_bound
            self.upper_bound = fit_upper_bound
            self.model_loaded = True
            
        except Exception as e:
            print(f"The following error occurred while loading the model: {e}.\nThe model was not loaded.")

        return None
    

    def _load_bounds(self, bounds: dict):
        try:
            fit_initial_guess = []
            fit_lower_bound = []
            fit_upper_bound = []

            for peak in bounds:
                amplitude = bounds[peak]['amplitude']['value']
                fwhm = bounds[peak]['fwhm']['value']
                eta = bounds[peak]['eta']['value']
            
                fit_initial_guess.extend([
                    peak,                              
                    fwhm,                                    
                    amplitude,                                
                    eta                                    
                ])
                
                fit_lower_bound.extend([
                    bounds[peak]['wavenumber']['min'], # bounds[peak]['wavenumber']['min'],
                    bounds[peak]["fwhm"]["min"],             
                    bounds[peak]["amplitude"]["min"],            
                    bounds[peak]['eta']['min']              
                ])
                
                fit_upper_bound.extend([
                    bounds[peak]['wavenumber']['max'],  
                    bounds[peak]["fwhm"]["max"],                  
                    bounds[peak]["amplitude"]["max"],             
                    bounds[peak]['eta']['max']
                ])

            if not self.model_loaded:
                self.initial_guess = fit_initial_guess
                self.model_loaded = True

            self.lower_bound = fit_lower_bound
            self.upper_bound = fit_upper_bound
            
        except Exception as e:
            print(f"The following error occurred while loading the bounds: {e}.\nThe bounds were not loaded.")
        return None


    def fit(self, x_values, y_values, peaks, param_dict=None, bounds=None, **kwargs):
        """
        Curve fitting using jaxfit.
        -------------------------------------
        Arguments:
            x_values - wavenumbers
            y_values - absorbance
            peaks - peak indices
            kwargs -- keyword arguments for curve_fit
        
        Returns:
            fit parameters
        """
        self.x_values = x_values
        self.y_values = y_values
        self.peaks = np.array([np.argmin(np.abs(self.x_values - peak)) for peak in peaks])

        if not self.model_loaded and bounds is None:
            self._create_params(
                self.x_values, 
                self.y_values, 
                self.peaks,
                param_dict=param_dict
            )
        if bounds is not None:
            self._load_bounds(bounds)

        jcf = CurveFit()

        bounds = (self.lower_bound, self.upper_bound)

        params_array, pcov = jcf.curve_fit(
            self.combined_pseudo_voigt,
            self.x_values,
            self.y_values,
            p0=self.initial_guess,
            bounds=bounds,
            **kwargs
            )
        
        self.params_array = params_array
        self.pcov = pcov
        self.predicted = np.array(self.combined_pseudo_voigt(self.x_values, *params_array))
        self.discrepancy = np.sqrt(np.mean((self.predicted - self.y_values)**2))
        self.r2 = r2_score(self.y_values, self.predicted)

        height = []
        absolute_component_areas = []
        relative_areas = []

        for i in range(0, len(params_array), 4):
            y_pred = self.combined_pseudo_voigt(self.x_values, *params_array[i:i+4])
            height.append(max(y_pred))

            abs_area = np.trapz(y_pred, x=self.x_values)
            absolute_component_areas.append(abs_area)

        total_absolute_area = np.sum(absolute_component_areas)

        if total_absolute_area > 1e-12:
            for abs_area in absolute_component_areas:
                relative_areas.append(abs_area / total_absolute_area)
        else:
            relative_areas = [0.0] * len(absolute_component_areas)

        self.wavenumbers = params_array[::4]
        self.gamma = params_array[1::4]
        self.amplitude = params_array[2::4]
        self.eta = params_array[3::4]
        self.areas = np.array(relative_areas)
        self.areas_abs = np.array(absolute_component_areas)
        self.height = np.array(height)
        self.wavenumbers_index = [np.argmin(np.abs(self.x_values - peak)) for peak in self.wavenumbers]

        self.params = pd.DataFrame({
            "wavenumber": self.wavenumbers,
            "FWHM": self.gamma,
            "amplitude": self.amplitude,
            "eta": self.eta,
            "height": self.height,
            "area": self.areas,
            "area_abs": self.areas_abs
        })

        return self.params

    def plot_fit(self, kind='fit', *, bin_num=200):
        """
        Plot resulting fit/residuals
        """
        if self.params is not None:
            if kind == 'fit':
                fig, ax = plt.subplots(figsize=(7, 3))

                ax.plot(self.x_values, self.predicted, label='Best fit', color='#d62728')
                ax.plot(self.x_values, self.y_values, label='Absorbance', color='#1f77b4')
                # ax.scatter(self.x_values[self.peaks], self.y_values[self.peaks], label='Peaks', color='red', s=10)

                label_text = f"DIS: {self.discrepancy:.1e}\n$R^2$: {self.r2:.5f}"

                x_text = 0.03
                y_text = 0.80
                ax.text(x_text, y_text, label_text, 
                        transform=ax.transAxes, ha='left', va='top', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

                for index, row in self.params.iterrows():
                    pseudo_voigt = self.combined_pseudo_voigt(self.x_values, 
                                                            row["wavenumber"], 
                                                            row["FWHM"], 
                                                            row["amplitude"], 
                                                            row["eta"])
                    ax.plot(self.x_values, pseudo_voigt, linestyle='--', linewidth=0.8, color='#7f7f7f')

                # ax.scatter(self.x_values[self.wavenumbers_index], 
                #            self.y_values[self.wavenumbers_index],
                #            marker='|',
                #            color='k',
                #            zorder=3
                #            )
                    
                ax.set_xlabel('Wavenumbers cm$^{-1}$')
                ax.set_ylabel('Absorbance')
                ax.legend()

            elif kind == 'residuals':
                fig, ax = plt.subplots(figsize=(7, 3))

                residual = self.y_values - self.predicted
                ax.plot(self.x_values, residual, color='#1f77b4', linewidth=0.8)

                ax.set_xlabel('Wavenumbers cm$^{-1}$')
                ax.set_ylabel('Residuals')

            elif kind == 'residuals_cumsum':
                fig, ax = plt.subplots(figsize=(7, 3))

                residuals = self.y_values - self.predicted
                bin_edges = np.linspace(self.x_values.min(), self.x_values.max(), num=bin_num)

                bin_indices = np.digitize(self.x_values, bins=bin_edges)
                binned_residuals = [residuals[bin_indices == i].sum() for i in range(1, len(bin_edges))]

                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                ax.bar(bin_centers, 
                       binned_residuals, 
                       width=np.diff(bin_edges), 
                       align='center', 
                       color='skyblue', 
                       edgecolor='k',
                       linewidth=0.5,
                       label="Binned Residuals Sum")
                
                ax.set_xlabel('Wavenumbers cm$^{-1}$')
                ax.set_ylabel('Residuals')

            else:
                raise ValueError('Kind should be either "fit" or "residuals"')
        else:
            raise ValueError("You need to fit a model first.")
    
        return fig, ax
    
    def plot_fit_plotly(self, kind='fit'):
        """
        Plot resulting fit/residuals
        """
        if self.params is not None:
            if kind == 'fit':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=self.x_values, y=self.y_values, mode='lines', name='Absorbance'))
                fig.add_trace(go.Scatter(x=self.x_values, y=self.predicted, mode='lines', name='Best fit', line=dict(color='black')))
                fig.add_trace(go.Scatter(
                    x=self.x_values[self.peaks],
                    y=self.y_values[self.peaks],
                    mode='markers',
                    name='Peaks',
                    marker=dict(size=8, color='red'),
                    visible='legendonly'
                ))

                fig.update_layout(title='Pseudo Voigt fit', 
                      xaxis_title='Wavenumbers', 
                      yaxis_title='Absorbance', 
                      height=650, 
                      width=950
                     )


                for index, row in self.params.iterrows():
                    pseudo_voigt = self.combined_pseudo_voigt(self.x_values, 
                                                            row["wavenumber"], 
                                                            row["FWHM"], 
                                                            row["amplitude"], 
                                                            row["eta"])
                    fig.add_trace(go.Scatter(x=self.x_values, 
                                 y=pseudo_voigt, 
                                 mode='lines', 
                                 line=dict(width=1, dash='dash'),
                                 name=str(round(row['wavenumber'], 2))))

            elif kind == 'residuals':
                residual = self.y_values - self.predicted
    
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(x=self.x_values, y=residual, mode='lines', name='Residuals', line=dict(color='green')))
                fig_residuals.update_layout(title='Pseudo Voigt fit Residuals', 
                        xaxis_title='Wavenumbers', 
                        yaxis_title='Residual', 
                        height=650, 
                        width=950
                        )
            
            else:
                raise ValueError('Kind should be either "fit" or "residuals"')
        else:
            raise ValueError("You need to fit a model first.")
    
        return fig
    
def combined_pseudo_voigt(x, *params):
    """ 
    Voigt profile approximation by 
    linear combination of Lorenzian and Gaussian distributions. 
    ------------------------------------------------------------
    Arguments:
        params - voigt profile parameters 
        (mu, gamma, amplitude, eta)
        mu - center of the distribution,
        gamma - FWHM for both gaussian and lorentzian terms,
        amplitude - peak height,
        eta - mixing parameter of the gaussian and lorentzian term

    Returns:
        jnp array
    """
    N = len(params) // 4
    result = np.zeros_like(x)
    for i in range(N):
        mu, gamma, amplitude, eta = params[i*4:(i+1)*4]

        a_G = (2 / gamma) * np.sqrt(np.log(2) / np.pi)
        b_G = (4 * np.log(2)) / (gamma**2)
        gaussian_term = a_G * np.exp(-b_G * (x - mu)**2)

        lorentzian_term = (1 / np.pi) * ((gamma / 2) / ((x - mu)**2 + (gamma / 2)**2))

        result += amplitude * (eta * gaussian_term + (1 - eta) * lorentzian_term)

    return result

def calculate_FWHM_pseudo_voigt(x, y):
    """
    Calculates FWHM of a given spectra
    ----------------------------------
    Arguments:
        x - wavenumbers
        y - absorbance

    Returns:
        FWHM
    """

    max_index = np.argmax(y)
    max_y = y[max_index]

    half_max = max_y / 2

    left_index = np.argmin(np.abs(y[:max_index] - half_max))
    right_index = max_index + np.argmin(np.abs(y[max_index:] - half_max))

    x_at_half_max_left = np.interp(half_max, y[left_index:left_index+2], x[left_index:left_index+2])
    x_at_half_max_right = np.interp(half_max, y[right_index:right_index+2], x[right_index:right_index+2])

    FWHM = x_at_half_max_right - x_at_half_max_left

    return FWHM