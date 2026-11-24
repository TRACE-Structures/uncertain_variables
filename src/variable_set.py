""" Represent a set of simulation variables with associated probability distributions and units. """

import numpy as np

from .variable import Variable

# from scipy.stats.qmc import Halton
# from scipy.stats.qmc import LatinHypercube as LHS
# from scipy.stats.qmc import Sobol
# from SALib.sample import saltelli
from SALib.sample import sobol 

class VariableSet():
    """ VariableSet class for managing a set of simulation variables.
        
        Attributes
        ----------
        normalized : bool
            Flag indicating whether the variables are normalized.

        variables : dict
            Dictionary of variable names and their corresponding Variable objects."""
        
    def __init__(self, variables=None, normalized=True):
        """ Initialize an empty VariableSet.

            Parameters
            ----------
            variables : list of Variable, optional
                List of Variable objects to initialize the VariableSet with. Default is None.

            normalized : bool, optional
                Flag indicating whether the variables are normalized. Default is True."""
        

        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean")
        
        self.normalized = normalized
        self.variables = {}
        if variables is not None:
            for variable in variables:
                self.add(variable)

    def __repr__(self):
        """ Returns the string representation of the VariableSet object.
        
            Returns
            -------
            repr_string : str
                String representation of the VariableSet object."""

        repr_str = f"VariableSet(normalized={self.normalized}, variables={list(self.get_variable_names())})"
        return repr_str

    def __eq__(self, other):
        """ Check if two VariableSet objects are equal.

            Parameters
            ----------
            other : VariableSet
                The other VariableSet object to compare with.

            Returns
            -------
            is_equal : bool
                True if the VariableSet objects are equal, False otherwise."""

        if not isinstance(other, VariableSet):
            return False
        return self.normalized == other.normalized and self.variables == other.variables

    def __len__(self):
        """ Returns the number of variables in the VariableSet.

            Returns
            -------
            num_variables : int
                The number of variables in the VariableSet."""

        num_variables = len(self.variables)
        return num_variables

    def __contains__(self, variable):
        """ Check if a variable with the given name exists in the VariableSet.

            Parameters
            ----------
            variable : Variable
                The Variable object to check for existence.

            Returns
            -------
            exists : bool
                True if the variable exists in the VariableSet, False otherwise."""

        if not isinstance(variable, Variable):
            raise TypeError("variable must be an instance of Variable")

        exists = self.variables.get(variable.name) == variable
        return exists

    def __iter__(self):
        """ Returns an iterator over the variable names in the VariableSet.

            Returns
            -------
            iterator : iterator
                An iterator over the variable names in the VariableSet."""

        iterator = iter(self.variables.values())
        return iterator

    def add(self, variable):
        """ Add a new Variable to the VariableSet.

            Parameters
            ----------
            variable : Variable 
                The Variable object to be added to the VariableSet."""

        if not isinstance(variable, Variable):
            raise TypeError("variable must be an instance of Variable")
        if variable.name in self.variables:
            raise KeyError(f"Variable name {variable.name} already exists in VariableSet. Try to rename the variable before adding it.")

        self.variables[variable.name] = variable

    def remove(self, variable_name):
        """ Remove a variable by name.

            Parameters
            ----------
            variable_name : str
                The name of the variable to be removed from the VariableSet."""
        
        if variable_name not in self.variables:
            raise KeyError(f"Variable name {variable_name} not found in VariableSet")

        del self.variables[variable_name]

    def update(self, variables):
        """ Add or replace variables from another VariableSet or dictionary.

            Parameters
            ----------
            variables : Variable, VariableSet or dict
                The source of variables to add or replace in the current VariableSet.
                If a dictionary is provided, the keys should be variable names and the values should be Variable objects."""

        if isinstance(variables, Variable):
            updates = {variables.name: variables}
        elif isinstance(variables, VariableSet):
            updates = {
                variable.name: variable
                for variable in variables.variables.values()
            }
        elif isinstance(variables, dict):
            updates = {}
            for name, variable in variables.items():
                if not isinstance(variable, Variable) or not isinstance(name, str):
                    raise TypeError("All keys must be strings and all values in the dictionary must be instances of Variable")
                updates[variable.name] = variable
        else:
            raise TypeError("variables must be an instance of Variable, VariableSet or a dictionary")

        self.variables.update(updates)

    def rename(self, old_names, new_names):
        """ Rename a variable or variables in the VariableSet.

            Parameters
            ----------
            old_names : str or list of str
                The current name(s) of the variable(s) to be renamed.
            new_names : str or list of str
                The new name(s) for the variable(s).

            Raises
            ------
            KeyError
                If the variable with old_name is not found in the VariableSet.
            ValueError
                If a variable with new_name already exists in the VariableSet."""

        if isinstance(old_names, str):
            old_names = [old_names]
        if isinstance(new_names, str):
            new_names = [new_names]

        if len(old_names) != len(new_names):
            raise ValueError("old_names and new_names must have the same length")
        if len(set(old_names)) != len(old_names):
            raise ValueError("old_names must not contain duplicates")
        if len(set(new_names)) != len(new_names):
            raise ValueError("new_names must not contain duplicates")

        for old_name, new_name in zip(old_names, new_names):
            if old_name not in self.variables:
                raise KeyError(f"Variable name {old_name} not found in VariableSet")
            if new_name in self.variables:
                raise ValueError(f"Variable name {new_name} already exists in VariableSet")

        for old_name, new_name in zip(old_names, new_names):
            var = self.variables.pop(old_name)
            var.rename(new_name)
            self.variables[new_name] = var

    def copy(self):
        """ Create a deep copy of the VariableSet.

            Returns
            -------
            VariableSet
                A new VariableSet instance with copies of all variables in the current VariableSet."""

        new_vs = VariableSet(normalized=self.normalized)
        for var in self.variables.values():
            new_vs.add(variable=var.copy())
        return new_vs

    # def variable_names(self):
    def get_variable_names(self):
        """ Get the names of all variables in the VariableSet.
        
            Returns
            -------
            variable_names: array_like
                An array of variable names in the VariableSet."""

        variable_names = list(self.variables.keys())
        return variable_names

    def get_variables(self):
        """ Get the dictionary of variables in the VariableSet.
        
            Returns
            -------
            variables: dict
                Dictionary of variable names and their corresponding Variable objects."""
        
        variables = self.variables
        return variables

    # def num_variables(self):
    def get_num_variables(self):
        """ Get the number of variables in the VariableSet.
        
            Returns
            -------
            num_variables: int
                Number of variables in the VariableSet."""
        
        num_variables = len(self)
        return num_variables

    def get_dist_types(self):
        """ Get the distribution types for all variables in the VariableSet.

            Returns
            -------
            dist_types: dict
                A dictionary of distribution types for each variable in the VariableSet, where keys are variable names and values are distribution types."""

        dist_types = {name: var.get_dist_type() for name, var in self.variables.items()}
        return dist_types

    def get_dist_params(self):
        """ Get the distribution parameters for all variables in the VariableSet.

            Returns
            -------
            dist_params: dict
                A dictionary of distribution parameters for each variable in the VariableSet, where keys are variable names and values are distribution parameters."""

        dist_params = {name: var.get_dist_params() for name, var in self.variables.items()}
        return dist_params

    def pdf(self, x, marginal=False):
        """ Return the joint probability density function values for all variables in the VariableSet at given points, 
            assuming that the variables are independent.

            Parameters
            ----------
            x : numpy.ndarray
                Array of points at which to evaluate the pdf. Shape should be (m, n) where m is the number of variables
                and n is the number of points.
            marginal : bool
                If True, return the marginal pdfs for each variable. Default is False.

            Returns
            -------
            y : numpy.ndarray
                Array of PDF values at the given points. Shape is (1, n) if marginal is False, otherwise (m, n)."""

        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy.ndarray")
        if x.ndim != 2:
            raise ValueError("x must have shape (num_variables, num_points)")
        if not isinstance(marginal, bool):
            raise ValueError("marginal must be a boolean")

        m = len(self)
        n = x.shape[1]
        if not x.shape[0] == m:
            raise ValueError("x must have shape (num_variables, num_points)")

        variables = self.variables
        if marginal:
            y = np.ones([m, n])
            for i, dist in enumerate(variables.values()):
                y[i, :] = dist.pdf(x[i])
        else:
            y = np.ones([1, n])
            for i, dist in enumerate(variables.values()):
                y = y * dist.pdf(x[i])
        return y

    def logpdf(self, x, marginal=False):
        """ Return the log joint probability density function values for all variables in the VariableSet at given points,
            assuming that the variables are independent.

            Parameters
            ----------
            x : numpy.ndarray
                Array of points at which to evaluate the logpdf. Shape should be (m, n) where m is the number of variables
                and n is the number of points.
            marginal : bool
                If True, return the marginal logpdf for each variable. If False, return the joint logpdf. Default is False.

            Returns
            -------
            y : numpy.ndarray
                Array of log PDF values at the given points. Shape is (1, n) if marginal is False, otherwise (m, n)."""

        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy.ndarray")
        if x.ndim != 2:
            raise ValueError("x must have shape (num_variables, num_points)")
        if not isinstance(marginal, bool):
            raise ValueError("marginal must be a boolean")

        m = len(self)
        n = x.shape[1]
        if not x.shape[0] == m:
            raise ValueError("x must have shape (num_variables, num_points)")

        variables = self.variables
        if marginal:
            y = np.empty((m, n))
            for i, variable in enumerate(variables.values()):
                y[i, :] = variable.logpdf(x[i])
        else:
            y = np.zeros((1, n))
            for i, variable in enumerate(variables.values()):
                y[0, :] += variable.logpdf(x[i])
        return y

    def cdf(self, x, marginal=False):
        """ Return the joint cumulative distribution function values for all variables in the VariableSet at given points,
            assuming that the variables are independent.

            Parameters
            ----------
            x : numpy.ndarray
                Array of points at which to evaluate the logpdf. Shape should be (m, n) where m is the number of variables
                and n is the number of points.
            marginal : bool
                If True, return the marginal logpdf for each variable. If False, return the joint logpdf. Default is False.

            Returns
            -------
            y : numpy.ndarray
                Array of CDF values at the given points. Shape is (1, n) if marginal is False, otherwise (m, n)."""

        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy.ndarray")
        if x.ndim != 2:
            raise ValueError("x must have shape (num_variables, num_points)")
        if not isinstance(marginal, bool):
            raise ValueError("marginal must be a boolean")

        m = len(self)
        n = x.shape[1]
        if not x.shape[0] == m:
            raise ValueError("x must have shape (num_variables, num_points)")

        variables = self.variables
        if marginal:
            y = np.empty((m, n))
            for i, variable in enumerate(variables.values()):
                y[i, :] = variable.cdf(x[i])
        
        else:
            y = np.ones([1, n])
            for i, dist in enumerate(variables.values()):
                y = y * dist.cdf(x[i])
        return y

    def ppf(self, y):
        """ Return the percentage point function (inverse of CDF) of all variables in the VariableSet at given probabilities,
            assuming that the variables are independent.

            Parameters
            ----------
            y : numpy.ndarray
                Array of probabilities at which to evaluate the PPF. Shape should be (m, n) where m is the number of variables
                and n is the number of points.

            Returns
            -------
            x : numpy.ndarray
                Array of PPF values at the given probabilities.
                Shape should be (m, n) where m is the number of variables and n is the number of points."""
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy.ndarray")
        if y.ndim != 2:
            raise ValueError("y must have shape (num_variables, num_points)")

        m = len(self)
        n = y.shape[1]
        if not y.shape[0] == m:
            raise ValueError("y must have shape (num_variables, num_points)")

        x = np.empty((m, n))
        for i, variable in enumerate(self.variables.values()):
            x[i, :] = variable.ppf(y[i])
        return x

    def mean(self):
        """ Return the mean values of all variables in the VariableSet.

            Returns
            -------
            means : dict
                Dictionary of mean values for each variable, where the keys are variable names and the values are the mean values."""
        means = {name: var.mean() for name, var in self.variables.items()}
        return means

    def var(self):
        """ Return the variance values of all variables in the VariableSet.

            Returns
            -------
            variances : dict
                Dictionary of variance values for each variable, where the keys are variable names and the values are the variance values."""
        variances = {name: var.var() for name, var in self.variables.items()}
        return variances

    def skew(self):
        """ Return the skewness values of all variables in the VariableSet.

            Returns
            -------
            skews : dict
                Dictionary of skewness values for each variable, where the keys are variable names and the values are the skewness values."""
        skews = {name: var.skew() for name, var in self.variables.items()}
        return skews

    def kurt(self):
        """ Return the kurtosis values of all variables in the VariableSet.

            Returns
            -------
            kurts : dict
                Dictionary of kurtosis values for each variable, where the keys are variable names and the values are the kurtosis values."""
        kurts = {name: var.kurt() for name, var in self.variables.items()}
        return kurts

    def moments(self):
        """ Return the first four central moments of all variables in the VariableSet.

            Returns
            -------
            moments : dict
                Dictionary of moments for each variable, where the keys are variable names and the values are tuples of the first four central moments (mean, variance, skewness, kurtosis)."""
        moments = {name: var.moments() for name, var in self.variables.items()}
        return moments

    # def sample(self, n, method='MC', random_seed=None, **kwargs):
    #     """ Generate samples from the VariableSet using specified sampling method.
            
    #         Parameters
    #         ----------
    #         n: int
    #             Number of samples to generate.

    #         method: {'MC', 'QMC_Halton', 'QMC_LHS', 'QMC_Sobol', 'Sobol_saltelli'}, default='MC'
    #             Sampling method to use. Options: 'MC' (Monte Carlo), 'QMC_Halton' (Quasi-Monte Carlo Halton), 
    #             'QMC_LHS' (Quasi-Monte Carlo Latin Hypercube), 'QMC_Sobol' (Quasi-Monte Carlo Sobol), 
    #             'Sobol_saltelli' (Sobol sampling using SALib Saltelli method). Default is 'MC'.

    #         random_seed: int or None
    #             Random seed for reproducibility. Default is None.

    #         **kwargs:
    #             Additional keyword arguments for specific sampling methods.
                
    #         Returns
    #         -------
    #         q_i_k: numpy.ndarray
    #             Array of shape (n, m) containing the generated samples in parameter space."""               
        
    #     m = self.num_variables()
    #     q_i_k = np.zeros([n, m])
    #     variables = self.variables
    #     if method == 'MC':
    #         np.random.seed(random_seed)
    #         xi = np.random.rand(n,m)
    #     # QMC methods
    #     elif method == 'QMC_Halton':
    #         gen = Halton(n, seed=random_seed)
    #         xi = gen.random(m).T
    #     elif method == 'QMC_LHS':
    #         sampler = LHS(d=n, seed=random_seed)
    #         xi = sampler.random(m).T
    #     elif method == 'QMC_Sobol':
    #         sampler = Sobol(d=n, seed=random_seed)
    #         xi = sampler.random(m).T
    #     elif method == 'Sobol_saltelli':
    #         problem = {'num_vars': m, 'names': self.variable_names(), 'dists': self.get_dist_types(), 'bounds': self.get_dist_params()}
    #         xi = saltelli.sample(problem, n)
    #         return xi
            
    #     for i, dist in enumerate(variables.values()):
    #         q_i_k[:, i] = dist.invcdf(xi[:, i])
            
    #     return q_i_k

    def sample(self, n, method="MC", seed=None): 
        """ Generate random samples from the variable's distribution using the specified sampling method.

            Parameters
            ----------
            n : int
                Number of samples to generate.

            method : str, default: "MC"
                Sampling method to use. Options are "MC" (Monte Carlo), "QMC_Halton" (Quasi-Monte Carlo using Halton sequence),
                "QMC_LHS" (Quasi-Monte Carlo using Latin Hypercube Sampling), and "QMC_Sobol" (Quasi-Monte Carlo using Sobol sequence).

            seed : int or None, default: None
                Seed for the random number generator.

            Returns
            -------
            samples : array_like
                Generated random samples from the variable."""

        m = len(self)
        if (
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n <= 0
        ):
            raise ValueError("Number of samples must be a positive integer.")
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
        ):
            raise ValueError("seed must be an integer or None.")
        samples = np.zeros([m, n])
        seeds = (
            np.random.SeedSequence(seed).generate_state(m)
            if seed is not None
            else [None] * m
        )
        for i, var in enumerate(self.variables.values()):
            samples[i, :] = var.sample(n, method=method, seed=seeds[i])
        return samples

    def sample_saltelli(self, n, calc_second_order=True, scramble=True, seed=None):
        """ Generate random samples using the Saltelli sampling method.

            Parameters
            ----------
            n : int
                Number of samples to generate.

            calc_second_order : bool, default: True
                Whether to calculate second-order effects.

            scramble : bool, default: True
                Whether to scramble the Sobol sequence.

            seed : int or None, default: None
                Seed for the random number generator.

            Returns
            -------
            samples : array_like
                Generated random samples from the variable set using the Saltelli method."""

        problem = {
            "num_vars": len(self),
            "names": self.get_variable_names(),
            "bounds": [[0, 1]] * len(self)
        }

        unit_samples = sobol.sample(problem, n, calc_second_order=calc_second_order, scramble=scramble, seed=seed)
        samples = np.zeros_like(unit_samples)
        for i, var in enumerate(self.variables.values()):
            samples[:, i] = var.ppf(unit_samples[:, i])
        return samples

    def get_bounds(self):
        """ Return the bounds of all variables in the VariableSet.

            Returns
            -------
            bounds : dict
                Dictionary where keys are variable names and values are arrays of shape (2,) containing the lower and upper bounds for each variable in the VariableSet."""

        bounds = {name: dist.get_bounds() for name, dist in self.variables.items()}
        return bounds

    def get_gpc_polysys(self):
        """ Return the GPC polynomial system for all variables in the VariableSet.
            
            Returns
            -------
            polysys : dict
                Dictionary where keys are variable names and values are the GPC polynomial systems for each variable."""

        polysys = {name: dist.get_gpc_polysys() for name, dist in self.variables.items()}
        return polysys
    
    def get_gpc_syschars(self):
        """ Return the GPC polynomial system characteristic string for all variables in the VariableSet.
            
            Returns
            -------
            syschars : dict
                Dictionary where keys are variable names and values are the GPC polynomial system characteristic strings for each variable."""
        
        syschars = {
            name: variable.get_gpc_syschar(self.normalized)
            for name, variable in self.variables.items()
        }
        return syschars

    def stdnor_to_germ(self, x):
        """ Convert from standard normal space to germ space for all variables in the VariableSet.

            Parameters
            ----------
            x : numpy.ndarray
                Points in standard normal space to be converted. Shape is (m, n) where m is the number of variables and n is the number of points.
                
            Returns
            -------
            y : numpy.ndarray
                Points in germ space. Shape is (m, n)."""

        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy.ndarray")
        if x.ndim != 2:
            raise ValueError("x must have shape (num_variables, num_points)")

        m = len(self)
        n = x.shape[1]
        if not x.shape[0] == m:
            raise ValueError("x must have shape (num_variables, num_points)")

        y = np.empty((m, n))
        for i, variable in enumerate(self.variables.values()):
            y[i, :] = variable.stdnor_to_germ(x[i])
        return y

    def germ_to_stdnor(self, y):
        """ Convert from germ space to standard normal space for all variables in the VariableSet.

            Parameters
            ----------
            y : numpy.ndarray
                Points in germ space to be converted. Shape is (m, n) where m is the number of variables and n is the number of points.
                
            Returns
            -------
            x : numpy.ndarray
                Points in standard normal space. Shape is (m, n)."""

        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy.ndarray")
        if y.ndim != 2:
            raise ValueError("y must have shape (num_variables, num_points)")

        m = len(self)
        n = y.shape[1]
        if not y.shape[0] == m:
            raise ValueError("y must have shape (num_variables, num_points)")

        x = np.empty((m, n))
        for i, variable in enumerate(self.variables.values()):
            x[i, :] = variable.germ_to_stdnor(y[i])
        return x
    
    def germ_to_dist(self, y):
        """ Convert from germ space to distribution space.

            Parameters
            ----------
            y : numpy.ndarray
                Points in germ space to be converted. Shape is (m, n) where m is the number of variables and n is the number of points.
                
            Returns
            -------
            x : numpy.ndarray
                Points in distribution space. Shape is (m, n)."""
        
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy.ndarray")
        if y.ndim != 2:
            raise ValueError("y must have shape (num_variables, num_points)")

        m = len(self)
        n = y.shape[1]
        if not y.shape[0] == m:
            raise ValueError("y must have shape (num_variables, num_points)")

        x = np.empty((m, n))
        for i, variable in enumerate(self.variables.values()):
            x[i, :] = variable.germ_to_dist(y[i])
        return x

    def dist_to_germ(self, x):
        """ Convert from distribution space to germ space.
            Parameters
            ----------
            x : array_like
                Points in distribution space to be converted. Shape is (m, n) where m is the number of variables and n is the number of points.
                
            Returns
            -------
            y : numpy.ndarray
                Points in germ space. Shape is (m, n)."""
        
        if not isinstance(x, np.ndarray):
            raise ValueError("x must be a numpy.ndarray")
        if x.ndim != 2:
            raise ValueError("x must have shape (num_variables, num_points)")

        m = len(self)
        n = x.shape[1]
        if not x.shape[0] == m:
            raise ValueError("x must have shape (num_variables, num_points)")

        y = np.empty((m, n))
        for i, variable in enumerate(self.variables.values()):
            y[i, :] = variable.dist_to_germ(x[i])
        return y

    def filter(self, condition):
        """ Create a new VariableSet containing only the variables that satisfy the given condition.

            Parameters
            ----------
            condition: callable
                A function that takes a variable as input and returns True if the variable should be included.
                
            Returns
            -------
            new_variable_set: VariableSet object
                New VariableSet containing only the variables that satisfy the condition."""

        new_variable_set = VariableSet(normalized=self.normalized)
        new_variables = {}
        for name, variable in self.variables.items():
            if condition(variable):
                new_variables[name] = variable
        new_variable_set.variables = new_variables
        return new_variable_set

    def select(self, variables, by="name"):
        """ Create a new VariableSet containing only the specified variables.

            Parameters
            ----------
            variables: list
                List of variable names or indices to include in the new VariableSet.
            by: str, optional
                Specify whether to select by "name" or "index". Default is "name".
                
            Returns
            -------
            new_variable_set: VariableSet object
                New VariableSet containing only the specified variables."""

        new_variable_set = VariableSet(normalized=self.normalized)
        new_variables = {}
        if by == "name":
            for name in variables:
                if name not in self.variables:
                    raise KeyError(f"Variable {name} not found in VariableSet")
                new_variables[name] = self.variables[name]
        elif by == "index":
            original_names = list(self.variables.keys())
            for idx in variables:
                if idx < 0 or idx >= len(original_names):
                    raise IndexError(f"Index {idx} out of range")
                name = original_names[idx]
                new_variables[name] = self.variables[name]
        else:
            raise ValueError("by must be either 'name' or 'index'")

        new_variable_set.variables = new_variables
        return new_variable_set

    def without(self, variables, by="name"):
        """ Create a new VariableSet excluding the specified variables.

            Parameters
            ----------
            variables: list
                List of variable names or indices to exclude from the new VariableSet.
            by: str, optional
                Specify whether to exclude by "name" or "index". Default is "name".
                
            Returns
            -------
            new_variable_set: VariableSet object
                New VariableSet excluding the specified variables."""

        new_variable_set = VariableSet(normalized=self.normalized)
        new_variables = {}
        if by == "name":
            for name, variable in self.variables.items():
                if name not in variables:
                    new_variables[name] = variable
        elif by == "index":
            original_names = list(self.variables.keys())
            exclude_names = set()
            for idx in variables:
                if idx < 0 or idx >= len(original_names):
                    raise IndexError(f"Index {idx} out of range")
                exclude_names.add(original_names[idx])
            for name, variable in self.variables.items():
                if name not in exclude_names:
                    new_variables[name] = variable
        else:
            raise ValueError("by must be either 'name' or 'index'")

        new_variable_set.variables = new_variables
        return new_variable_set
    
    # def filter(self, variable_names):
    #     """ Create a new VariableSet containing only the specified variables.
            
    #         Parameters
    #         ----------  
    #         variable_names: list
    #             List of variable names to include in the new VariableSet.
                
    #         Returns
    #         -------
    #         new_variable_set: VariableSet object
    #             New VariableSet containing only the specified variables."""
        
    #     new_variable_set = VariableSet(normalized=self.normalized)
    #     new_variables = {}
    #     # To avoid confusion, use a list of original names to map indices
    #     original_names = list(self.variables.keys())
    #     for name in variable_names:
    #         if name not in self.variables:
    #             raise KeyError(f"Variable {name} not found in VariableSet")
    #         idx = original_names.index(name)
    #         new_variables[name] = self.variables[name]
    #     new_variable_set.variables = new_variables
    #     return new_variable_set

    # def diminished_variable_set(self, indices):
    #     """ Create a new VariableSet by selecting a subset of variables based on provided indices.
            
    #         Parameters
    #         ----------
    #         indices: list or array_like
    #             List of indices of variables to include in the new VariableSet.
                
    #         Returns
    #         -------
    #         new_variable_set: VariableSet object
    #             New VariableSet containing only the selected variables."""
        
    #     assert self.num_variables() >= len(indices)

    #     diminished_variable_names = np.array(self.variable_names())[indices]
    #     new_variable_set = VariableSet(normalized=self.normalized)
    #     new_variables = {}
    #     for i in range(len(indices)):
    #         new_variables[diminished_variable_names[i]] = self.variables[diminished_variable_names[i]]
    #     new_variable_set.variables = new_variables
    #     return new_variable_set