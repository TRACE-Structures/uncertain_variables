""" Represent a set of simulation variables with associated probability distributions. """

import numpy as np

from src.variable import Variable

from scipy.stats.qmc import Halton
from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol
# from SALib.sample import saltelli

class VariableSet():
    """ VariableSet class for managing a set of simulation variables.
        
        Attributes
        ----------
        normalized : bool
            Flag indicating whether the variables are normalized.

        variables : dict
            Dictionary of variable names and their corresponding Variable objects."""
        
    def __init__(self, normalized=True):
        """ Initialize an empty VariableSet.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether the variables are normalized. Default is True."""

        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean")
        
        self.normalized = normalized
        self.variables = {}

    def add(self, variable):
        """ Add a new Variable to the VariableSet.

            Parameters
            ----------
            variable : Variable 
                The Variable object to be added to the VariableSet."""

        if not isinstance(variable, Variable):
            raise TypeError("variable must be an instance of Variable")
        if variable.name in self.variables.keys():
            raise KeyError(f"Variable name {variable.name} already exists in VariableSet")

        variable_name = variable.name
        self.variables[variable_name] = variable

    def remove(self, variable_name):
        """ Remove a variable by name.

            Parameters
            ----------
            variable_name : str
                The name of the variable to be removed from the VariableSet."""
        
        if variable_name not in self.variables:
            raise KeyError(f"Variable name {variable_name} not found in VariableSet")

        del self.variables[variable_name]

    def get_dist_types(self, array_like=True):
        """ Get the distribution types for all variables in the VariableSet.

            Parameters
            ----------
            array_like : bool
                If True, return as a numpy array. 
                If False, returns as a dictionary, where the keys are variable names and the values are distribution types.
                Default is True.

            Returns
            -------
            dist_types : array_like or dict
                An array_like or dictionary of distribution types for each variable in the VariableSet."""

        if array_like:
            dist_types = [var.get_dist_type() for var in self.variables.values()]
        else:
            dist_types = {name: var.get_dist_type() for name, var in self.variables.items()}
        return dist_types
    
    def get_dist_params(self, array_like=True):
        """ Get the parameters of distributions for all variables in the VariableSet.

            Parameters
            ----------
            array_like : bool
                If True, return as a numpy array. 
                If False, returns as a dictionary, where the keys are variable names and the values are distribution parameters.
                Default is True.
            
            Returns
            -------
            dist_params: array_like or dict
                An array_like or dictionary of distribution parameters for each variable in the VariableSet."""
        
        if array_like:
            dist_params = [var.get_dist_params() for var in self.variables.values()]
        else:
            dist_params = {name: var.get_dist_params() for name, var in self.variables.items()}
        return dist_params

    # def num_variables(self):
    def get_num_variables(self):
        """ Get the number of variables in the VariableSet.
        
            Returns
            -------
            num_variables: int
                Number of variables in the VariableSet."""
        
        num_variables = len(self.variables)
        return num_variables

    # def variable_names(self):
    def get_variable_names(self):
        """ Get the names of all variables in the VariableSet.
        
            Returns
            -------
            variable_names: list
                List of variable names in the VariableSet."""
        
        variable_names = np.array(self.variables.keys())
        return variable_names
    
    def get_variables(self):
        """ Get the dictionary of variables in the VariableSet.
        
            Returns
            -------
            variables: dict
                Dictionary of variable names and their corresponding Variable objects."""
        
        variables = self.variables
        return variables

    def pdf(self, q):
        """ Get the probability density function values for all variables in the VariableSet at given points.
            
            Parameters
            ----------
            q: numpy.ndarray of shape (num_variables, num_points)
                Array of points at which to evaluate the pdf. Shape should be (m, n) where m is the number of variables
                and n is the number of points.
                
            Returns
            -------
            p_q: numpy.ndarray
                Array of pdf values at the given points. Shape is (1, n)."""   
        
        m = self.get_num_variables()
        assert(q.shape[0] == m)
        n = q.shape[1]

        variables = self.variables
        p_q = np.ones([1, n])
        for i, dist in enumerate(variables.values()):
            p_q = p_q * dist.pdf(q[i])
        return p_q
    
    def logpdf(self, q):
        """ Get the log probability density function values for all variables in the VariableSet at given points.
            
            Parameters
            ----------
            q: numpy.ndarray
                Array of points at which to evaluate the logpdf. Shape should be (m, n) where m is the number of variables
                and n is the number of points.
                
            Returns
            -------
            p_q: numpy.ndarray
                Array of logpdf values at the given points. Shape is (1, n)."""
        
        m = self.get_num_variables()
        assert(q.shape[0] == m)
        n = q.shape[1]

        variables = self.variables
        p_q = np.zeros([1, n])
        for i, dist in enumerate(variables.values()):
            p_q = p_q + dist.logpdf(q[i])
        return p_q

    def cdf(self, q):
        """ Get the cumulative distribution function values for all variables in the VariableSet at given points.
            
            Parameters
            ----------
            q: numpy.ndarray
                Array of points at which to evaluate the cdf. Shape should be (m, n) where m is the number of variables
                and n is the number of points.
                
            Returns
            -------
            p_q: numpy.ndarray
                Array of cdf values at the given points. Shape is (m, n)."""
        
        m = self.num_variables()
        assert (q.shape[0] == m)
        n = q.shape[1]

        variables = self.variables
        p_q = np.empty([m, n])
        for i, dist in enumerate(variables.values()):
            p_q[i,:] = dist.cdf(q[i,:])
        return p_q

    def mean(self, array_like=True):
        """ Get the mean values of all variables in the VariableSet.

            Parameters
            ----------
            array_like : bool
                If True, return as a numpy array. 
                If False, returns as a dictionary, where the keys are variable names and the values are mean values.
                Default is True.
            
            Returns
            -------
            means: array_like or dict
                An array_like or dictionary of mean values for each variable in the VariableSet."""
        
        # m = self.get_num_variables()
        # variables = self.variables
        # q_mean = np.zeros([m,1])
        # for i, dist in enumerate(variables.values()):
        #     q_mean[i] = dist.mean()
        # return q_mean

        if array_like:
            means = np.array([var.mean() for var in self.variables.values()])
        else:
            means = {name: var.mean() for name, var in self.variables.items()}
        return means

    def var(self, array_like=True):
        """ Get the variances of all variables in the VariableSet.

            Parameters
            ----------
            array_like : bool
                If True, return as a numpy array. 
                If False, returns as a dictionary, where the keys are variable names and the values are variances.
                Default is True.

            Returns
            -------
            vars: array_like or dict
                An array_like or dictionary of variances for each variable in the VariableSet."""

        # m = self.num_variables()
        # variables = self.variables
        # var = np.zeros([m, 1])
        # for i, dist in enumerate(variables.values()):
        #     var[i] = dist.var()
        # return var
    
        if array_like:
            vars = np.array([var.var() for var in self.variables.values()])
        else:
            vars = {name: var.var() for name, var in self.variables.items()}
        return vars

    def skew(self, array_like=True):
        """ Get the skewness of all variables in the VariableSet.

            Parameters
            ----------
            array_like : bool
                If True, return as a numpy array. 
                If False, returns as a dictionary, where the keys are variable names and the values are skewness values.
                Default is True.

            Returns
            -------
            skews: array_like or dict
                An array_like or dictionary of skewness values for each variable in the VariableSet."""

        if array_like:
            skews = np.array([var.skew() for var in self.variables.values()])
        else:
            skews = {name: var.skew() for name, var in self.variables.items()}
        return skews

    def kurt(self, array_like=True):
        """ Get the kurtosis of all variables in the VariableSet.

            Parameters
            ----------
            array_like : bool
                If True, return as a numpy array. 
                If False, returns as a dictionary, where the keys are variable names and the values are kurtosis values.
                Default is True.

            Returns
            -------
            kurts: array_like or dict
                An array_like or dictionary of kurtosis values for each variable in the VariableSet."""

        if array_like:
            kurts = np.array([var.kurt() for var in self.variables.values()])
        else:
            kurts = {name: var.kurt() for name, var in self.variables.items()}
        return kurts

    def get_gpc_syschars(self):
        """ Get the GPC polynomial system characteristic string for all variables in the VariableSet.
            
            Returns
            -------
            syschar: str
                GPC polynomial system characteristic string for the VariableSet."""
        
        variables = self.variables
        syschar = ''
        for i, dist in enumerate(variables.values()):
            syschar = syschar + dist.get_base_dist().orth_polysys_syschar(True)
        return syschar
    
    def germ2variable(self, xi_k_i):
        """ Convert from germ space to variable space.
            Parameters
            ----------
            xi_k_i : array_like
                Points in germ space to be converted. Shape is (n, m) where n is the number of points and m is the number of variables.
                
            Returns
            -------
            q_k_i : numpy.ndarray
                Points in variable space. Shape is (n, m)."""
        
        m = self.num_variables()
        assert (xi_k_i.shape[1] == m)
        n = xi_k_i.shape[0]

        variables = self.variables
        q_k_i = np.zeros([n, m])
        for i, dist in enumerate(variables.values()):
            q_k_i[:, i] = dist.base2dist(xi_k_i[:,i])
        return q_k_i
    
    def variable2germ(self, q_k_i):
        """ Convert from variable space to germ space.
            Parameters
            ----------
            q_k_i : array_like
                Points in variable space to be converted. Shape is (n, m) where n is the number of points and m is the number of variables.
                
            Returns
            -------
            xi_k_i : numpy.ndarray
                Points in germ space. Shape is (n, m)."""
        
        m = self.num_variables()
        assert (q_k_i.shape[1] == m)
        n = q_k_i.shape[0]

        variables = self.variables
        xi_k_i = np.zeros([n, m])
        for i, dist in enumerate(variables.values()):
            xi_k_i[:, i] = dist.dist2base(q_k_i[:,i])
        return xi_k_i
    
    def get_bounds(self):
        """ Get the bounds of all variables in the VariableSet.
            
            Returns
            -------
            bounds: numpy.ndarray
                Array of shape (m, 2) containing the lower and upper bounds for each variable in the VariableSet."""
        
        m = self.num_variables()
        bounds = np.zeros((m, 2))
        for i, dist in enumerate(self.variables.values()):
            bounds[i, :] = dist.get_bounds()
        return bounds

    def sample(self, n, method='MC', random_seed=None, **kwargs):
        """ Generate samples from the VariableSet using specified sampling method.
            
            Parameters
            ----------
            n: int
                Number of samples to generate.

            method: {'MC', 'QMC_Halton', 'QMC_LHS', 'QMC_Sobol', 'Sobol_saltelli'}, default='MC'
                Sampling method to use. Options: 'MC' (Monte Carlo), 'QMC_Halton' (Quasi-Monte Carlo Halton), 
                'QMC_LHS' (Quasi-Monte Carlo Latin Hypercube), 'QMC_Sobol' (Quasi-Monte Carlo Sobol), 
                'Sobol_saltelli' (Sobol sampling using SALib Saltelli method). Default is 'MC'.

            random_seed: int or None
                Random seed for reproducibility. Default is None.

            **kwargs:
                Additional keyword arguments for specific sampling methods.
                
            Returns
            -------
            q_i_k: numpy.ndarray
                Array of shape (n, m) containing the generated samples in parameter space."""               
        
        m = self.num_variables()
        q_i_k = np.zeros([n, m])
        variables = self.variables
        if method == 'MC':
            np.random.seed(random_seed)
            xi = np.random.rand(n,m)
        # QMC methods
        elif method == 'QMC_Halton':
            gen = Halton(n, seed=random_seed)
            xi = gen.random(m).T
        elif method == 'QMC_LHS':
            sampler = LHS(d=n, seed=random_seed)
            xi = sampler.random(m).T
        elif method == 'QMC_Sobol':
            sampler = Sobol(d=n, seed=random_seed)
            xi = sampler.random(m).T
        elif method == 'Sobol_saltelli':
            problem = {'num_vars': m, 'names': self.variable_names(), 'dists': self.get_dist_types(), 'bounds': self.get_dist_params()}
            xi = saltelli.sample(problem, n)
            return xi
            
        for i, dist in enumerate(variables.values()):
            q_i_k[:, i] = dist.invcdf(xi[:, i])
            
        return q_i_k
    
    def filter(self, variable_names):
        """ Create a new VariableSet containing only the specified variables.
            
            Parameters
            ----------  
            variable_names: list
                List of variable names to include in the new VariableSet.
                
            Returns
            -------
            new_variable_set: VariableSet object
                New VariableSet containing only the specified variables."""
        
        new_variable_set = VariableSet(normalized=self.normalized)
        new_variables = {}
        # To avoid confusion, use a list of original names to map indices
        original_names = list(self.variables.keys())
        for name in variable_names:
            if name not in self.variables:
                raise KeyError(f"Variable {name} not found in VariableSet")
            idx = original_names.index(name)
            new_variables[name] = self.variables[name]
        new_variable_set.variables = new_variables
        return new_variable_set

    def diminished_variable_set(self, indices):
        """ Create a new VariableSet by selecting a subset of variables based on provided indices.
            
            Parameters
            ----------
            indices: list or array_like
                List of indices of variables to include in the new VariableSet.
                
            Returns
            -------
            new_variable_set: VariableSet object
                New VariableSet containing only the selected variables."""
        
        assert self.num_variables() >= len(indices)

        diminished_variable_names = np.array(self.variable_names())[indices]
        new_variable_set = VariableSet(normalized=self.normalized)
        new_variables = {}
        for i in range(len(indices)):
            new_variables[diminished_variable_names[i]] = self.variables[diminished_variable_names[i]]
        new_variable_set.variables = new_variables
        return new_variable_set