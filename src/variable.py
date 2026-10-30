""" Represent random variables for uncertainty quantification."""

# import pint
import numpy as np

from src.distributions.Distribution import Distribution
# from src.polysys import syschar_to_polysys

class Variable:
    """ Represent a named variable with a distribution and an optional unit.

        Attributes
        ----------
            name : str
                Variable name.

            dist : Distribution object or None
                Distribution for a random variable, or None for a fixed variable.

            unit : str or None
                Unit string, or None if no unit is specified. """
        
    def __init__(self, name, dist, unit=None):
        """ Initialize a named variable with a distribution and an optional unit.

            Parameters
            ----------
            name : str
                Variable name.

            dist : Distribution 
                Distribution for a random variable, or value for a fixed variable.

            unit : str or None, optional
                Unit associated with the variable.

            Raises
            ------
            TypeError
                If ``dist`` is not a Distribution.
        """
        
        # self.name = name
        # self.dist = dist_or_num if hasattr(dist_or_num, 'pdf') else None
        # self.fixed_val = dist_or_num if not hasattr(dist_or_num, 'pdf') else None
        # self.is_fixed = self.fixed_val is not None
        # self.unit = unit

        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(dist, Distribution):
            raise TypeError("dist must be an instance of Distribution")
        if unit is not None and not isinstance(unit, str):
            raise TypeError("unit must be a string or None")
        
        self.name = name
        self.dist = dist
        self.unit = unit

    def __repr__(self):
        """ Returns the string representation of the Variable object.
        
            Returns
            -------
            repr_string : str
                String representation of the Variable object."""

        repr_string = 'Variable({}, {}, {})'.format(self.name, self.dist, self.unit)
        return repr_string

    def __eq__(self, other):
        """ Check equality of two Variable objects.

            Parameters
            ----------
            other : Variable
                Another Variable object to compare with.

            Returns
            -------
            is_equal : bool
                True if the variables are equal, False otherwise."""

        if not isinstance(other, Variable):
            return False
        
        is_equal = (self.name == other.name) and (self.unit == other.unit) and (self.dist == other.dist)
        return is_equal

    def get_dist_type(self):
        """ Return the type of the variable's distribution.

            Returns
            -------
            var_type : str
                The name identifying the variable's distribution."""

        var_type = self.dist.get_dist_type()
        return var_type


    def get_dist_params(self):
        """ Return the parameters of the variable's distribution.

            Returns
            -------
            var_params : array_like
                Parameters of the variable's distribution."""

        var_params = self.dist.get_dist_params()
        return var_params
    
    def pdf(self, x):
        """ Return the probability density function of the variable's distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the PDF.

            Returns
            -------
            y : array_like
                PDF values evaluated at x."""
        
        y = self.dist.pdf(x)
        return y

    def logpdf(self, x):
        """ Return the log of the probability density function of the variable's distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the log PDF.

            Returns
            -------
            y : array_like
                Log PDF values evaluated at x."""
        
        y = self.dist.logpdf(x)
        return y

    def cdf(self, x):
        """ Return the cumulative distribution function of the variable's distribution, evaluated at x.

            Parameters
            ----------
            x : array_like
                Points at which to evaluate the CDF.

            Returns
            -------
            y : array_like
                CDF values evaluated at x."""
        
        y = self.dist.cdf(x)
        return y
    
    def ppf(self, y):
        """ Return the percentage point function (inverse of cdf) of the variable's distribution, evaluated at y.

            Parameters
            ----------
            y: array_like
                Points at which to evaluate the PPF.

            Returns
            -------
            x : array_like
                PPF values evaluated at y."""

        x = self.dist.ppf(y)
        return x
        
    def mean(self):
        """ Return mean value of the variable's distribution.

            Returns
            -------
            mean : float
                Mean of the variable's distribution."""
        
        mean = self.dist.mean()
        return mean

    def var(self):
        """ Return variance of the variable's distribution.

            Returns
            -------
            var : float
                Variance of the variable's distribution."""

        var = self.dist.var()
        return var

    def skew(self):
        """ Return skewness of the variable's distribution.

            Returns
            -------
            skew : float
                Skewness of the variable's distribution."""

        skew = self.dist.skew()
        return skew
    
    def kurt(self):
        """ Return kurtosis of the variable's distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the variable's distribution."""

        kurt = self.dist.kurt()
        return kurt
    
    def moments(self):
        """ Return the moments of the variable's distribution.

            Returns
            -------
            moments : array_like
                Array containing the mean, variance, skewness, and kurtosis of the distribution."""

        moments = self.dist.moments()
        return moments

    # def sample(self, n):
    #     return self.dist.sample(n)

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
        
        samples = self.dist.sample(n, method=method, seed=seed)
        return samples

    def transform(self, shift=0, scale=1):
        """ Return a translated and scaled version of the variable's distribution.

            The transformation is defined as

            Y = scale * X + shift,

            where X is the variable's original distribution.
        
            Parameters
            ----------
            shift : float, default: 0
                Shift to apply to the distribution.

            scale : float, default: 1
                Positive scale factor to apply to the distribution.
            
            Returns
            -------
            new_dist : Distribution
                Transformed variable's distribution."""
        
        new_dist = self.dist.transform(shift=shift, scale=scale)
        return new_dist

    def get_shift(self):
        """ Return the shift (location) parameter of the variable's distribution.

            Returns
            -------
            shift : float
                Shift (location) parameter of the variable's distribution."""
        
        shift = self.dist.get_shift()
        return shift

    def get_scale(self):
        """ Return the scale parameter of the variable's distribution.

            Returns
            -------
            scale : float
                Scale parameter of the variable's distribution."""
        
        scale = self.dist.get_scale()
        return scale

    def fix_moments(self, mean, var, loc=None, scale=None):
        """ Return a transformed version of the variable's distribution with specified mean and variance.

            Parameters
            ----------
            mean : float
                Desired mean of the distribution.
                
            var : float
                Desired positive variance of the distribution.

            loc : float or None, default: None
                Desired location parameter of the distribution, distribution specific. 

            scale : float or None, default: None
                Desired scale parameter of the distribution, distribution specific.

            Returns
            -------
            new_dist : Distribution
                Variable's distribution with the specified mean and variance."""
        
        if loc is None and scale is None:
            new_dist = self.dist.fix_moments(mean=mean, var=var)
        elif loc is not None and scale is None:
            new_dist = self.dist.fix_moments(mean=mean, var=var, loc=loc)
        elif scale is not None and loc is None:
            new_dist = self.dist.fix_moments(mean=mean, var=var, scale=scale)
        else:
            new_dist = self.dist.fix_moments(mean=mean, var=var, loc=loc, scale=scale)
        return new_dist

    def get_bounds(self, delta=0.02):
        """ Return the bounds of the distribution.
            
            Parameters
            ----------
            delta : float, optional, default: 0.02
                Small probability to define the bounds.
            
            Returns
            -------
            bounds : array_like
                Bounds of the distribution as [lower_bound, upper_bound]."""
        
        bounds = self.dist.get_bounds(delta=delta)
        return bounds

    def fix_bounds(self, new_min, new_max, q0=0, q1=1):
        """ Fix the distribution to have specified bounds.
        
            Parameters
            ----------
            new_min : float
                Desired minimum of the distribution.
                
            new_max : float
                Desired maximum of the distribution.
                
            q0 : float, optional
                Quantile corresponding to the minimum (default is 0).
                
            q1 : float, optional
                Quantile corresponding to the maximum (default is 1).
                
            Returns
            -------
            new_dist : Distribution
                Translated and scaled distribution with specified bounds."""
        
        new_dist = self.dist.fix_bounds(new_min=new_min, new_max=new_max, q0=q0, q1=q1)
        return new_dist
    
    def get_gpc_dist(self):
        """ Return the GPC base distribution of the variable's distribution.

            Returns
            -------
            dist : Distribution object
                GPC base distribution of the variable's distribution."""
        
        dist_germ = self.dist.gpc_dist()
        return dist_germ

    def get_gpc_polysys(self):
        """ Return the GPC polynomial system of the variable's distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system.

            Returns
            -------
            polysys : object
                GPC polynomial system of the variable's distribution."""
        
        polysys = self.dist.gpc_polysys()
        return polysys

    def get_gpc_syschar(self, normalized):
        """ Return the GPC polynomial system characteristic string of the variable's distribution.

            Parameters
            ----------
            normalized : bool
                Flag indicating whether to return the normalized polynomial system characteristic string.

            Returns
            -------
            syschar : str
                GPC polynomial system characteristic string of the variable's distribution if not fixed, otherwise None."""
        
        syschar = self.dist.gpc_syschar(normalized)
        return syschar

    def stdnor_to_dist(self, x):
        """ Convert from standard normal space to the variable's distribution space.
        
            Parameters
            ----------
            x : array_like
                Points in standard normal space.
                
            Returns
            -------
            y : array_like
                Points in variable's distribution space."""
        
        y = self.dist.stdnor_to_dist(x)
        return y

    def dist_to_stdnor(self, y):
        """ Convert from variable's distribution space to standard normal space.
        
            Parameters
            ----------
            y : array_like
                Points in variable's distribution space.
                
            Returns
            -------
            x : array_like
                Points in standard normal space."""

        x = self.dist.dist_to_stdnor(y)
        return x

    def germ_to_dist(self, y):
        """ Convert from germ to variable space of the variable's distribution.

            Parameters
            ----------
            y : array_like
                Points in germ space.

            Returns
            -------
            x : array_like
                Points in variable space of the variable's distribution."""
        
        x = self.dist.germ_to_dist(y)
        return x

    def dist_to_germ(self, x):
        """ Convert from variable to germ space of the variable's distribution.

            Parameters
            ----------
            x : array_like
                Points in variable space.
            
            Returns
            -------
            y : array_like
                Points in germ space of the variable's distribution."""
        
        y = self.dist.dist_to_germ(x)
        return y

    def add_unit(self, unit):
        """ Add a unit to the variable if it doesn't have one.

            Parameters
            ----------
            unit : str
                Unit to be added.

            Raises
            ------
            ValueError
                If the variable already has a unit."""
        
        if self.unit is not None:
            raise ValueError("Variable already has a unit.")
        self.unit = unit
    
    def convert_to(self, to_unit):
        """ Convert the variable and its distribution parameters to a different unit.
            
            Parameters
            ----------
            to_unit : str
                Target unit.

            Returns
            -------
            report : str
                Description of the conversion.

            Raises
            ------
            ValueError
                If the variable has no source unit.
            pint.DimensionalityError
                If the source and target units are incompatible."""
        
        # ureg = pint.UnitRegistry()

        # if self.unit is None:
        #     raise ValueError("Current unit is not defined.")
        
        # if self.is_fixed:
        #     q = self.fixed_val * ureg(self.unit)
        #     self.fixed_val = q.to(to_unit).magnitude
        #     self.unit = to_unit
        
        # else:
        #     for attr, value in vars(self.dist).items():
        #         if isinstance(value, (int, float)) and not isinstance(value, bool):
        #             q = value * ureg(self.unit)
        #             setattr(self.dist, attr, q.to(to_unit).magnitude)
        #     self.unit = to_unit

        if not self.unit:
            raise ValueError("Variable has no source unit.")
            
        from_unit = self.unit
        report = self.dist.convert_to(from_unit=from_unit, to_unit=to_unit)

        self.unit = to_unit
        report = f"Variable {self.name}: {report}"
        return report