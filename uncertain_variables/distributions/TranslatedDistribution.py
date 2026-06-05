from .Distribution import Distribution

class TranslatedDistribution(Distribution):
    ''' Distribution obtained by applying an affine 
        transformation to another base distribution.
        
        Attributes
        ----------
        dist: Distribution
            Underlying base distribution.

        shift: float
            Additive translation applied after scaling.

        scale: float
            Scaling factor applied to centered values.

        center: float
            Center point for scaling transformation.
            
        Methods
        -------
        __init__(dist, shift, scale, center=None)
            Initialize the translated distribution.
            
        __repr__()
            Return a string representation of the translated distribution.
            
        get_dist_type()
            Return the type identifier of this distribution.
            
        get_dist_params()
            Return the parameter tuple of the translated distribution.
            
        translate_points(x, forward)
            Apply the forward or backward affine transformation to points.
            
        pdf(x)
            Evaluate the probability density function.

        cdf(x)
            Evaluate the cumulative distribution function.

        invcdf(y)
            Evaluate the inverse cumulative distribution function.

        mean()
            Return the mean of the translated distribution.

        var()
            Return the variance of the translated distribution.

        skew()
            Return the skewness of the translated distribution.

        kurt()
            Return the kurtosis of the translated distribution.

        sample(n)
            Draw random samples from the translated distribution.

        moments()
            Return the first four moments of the translated distribution.

        get_base_dist()
            Return the underlying base distribution.

        translate_points_forward(x, shift, scale, center)
            Apply the forward affine transformation.

        translate_points_backwards(x, shift, scale, center)
            Apply the inverse affine transformation.

        translate_moments(m, shift, scale, center)
            Transform the moments of a distribution under an affine transformation.'''
    
    def __init__(self, dist, shift, scale, center=None):
        ''' Initialize the translated distribution.
        
            Parameters
            ----------
            dist: Distribution
                Underlying base distribution
                
            shift: float
                Additive translation applied after scaling.
            
            scale: float
                Scaling factor applied to centered values.
            
            center: float
                Center point for scaling transformation.'''
        
        self.dist = dist
        self.shift = shift
        self.scale = scale
        if center is None:
            self.center = self.dist.moments()[0]
        else:
            self.center = center

    def __repr__(self):
        ''' Returns the string representation of the TranslatedDistribution object.
        
            Returns
            -------
            repr_string : str
                String representation of the TranslatedDistribution object.'''

        repr_string = "Translated({}, {}, {}, {})".format(
            self.dist, self.shift, self.scale, self.center
        )
        return repr_string
    
    def __eq__(self, other):
        ''' Check if two TranslatedDistribution objects are equal.

            Parameters
            ----------
            other: TranslatedDistribution
                Another TranslatedDistribution object to compare with.

            Returns
            -------
            bool
                True if the two distributions are equal, False otherwise.'''

        if not isinstance(other, TranslatedDistribution):
            return False
        is_equal = (self.dist == other.dist) and (self.shift == other.shift) and (self.scale == other.scale) and (self.center == other.center)
        return is_equal
    
    def get_dist_type(self):
        """ Return the type identifier of this distribution.

        Returns
        -------
        str
            Always returns ``'translated'``."""
        return "translated"
    
    def get_dist_params(self):
        """ Return the parameter tuple of the translated distribution.

            Returns
            -------
            parameters: tuple
                Parameters of the base distribution followed by
                ``(shift, scale, center)``."""

        parameters = self.dist.get_dist_params() + (self.shift, self.scale, self.center)
        return parameters

    def translate_points(self, x, forward):
        """ Apply the forward or backward affine transformation to points.

            Parameters
            ----------
            x : array_like
                Input values to transform.

            forward : bool
                If True, apply the forward transformation (base → translated).
                If False, apply the backward transformation (translated → base).

            Returns
            -------
            y: array_like
                Transformed points."""

        if forward:
            y = TranslatedDistribution.translate_points_forward(
                x, self.shift, self.scale, self.center
            )
        else:
            y = TranslatedDistribution.translate_points_backwards(
                x, self.shift, self.scale, self.center
            )
        return y

    def pdf(self, x):
        """ Evaluate the probability density function.

            Parameters
            ----------
            x: array_like
                Points at which to evaluate the PDF of the translated distribution.

            Returns
            -------
            y: array_like
                PDF values."""

        x = self.translate_points(x, False)
        y = self.dist.pdf(x) / self.scale
        return y

    def cdf(self, x):
        """ Evaluate the cumulative distribution function.

            Parameters
            ----------
            x: array_like
                Points at which to evaluate the CDF of the translated distribution.

            Returns
            -------
            y: array_like
                CDF values."""

        x = self.translate_points(x, False)
        y = self.dist.cdf(x) / 1
        return y

    def invcdf(self, y):
        """ Evaluate the inverse cumulative distribution function.

            Parameters
            ----------
            y: array_like
                CDF values for which to compute the quantiles.

            Returns
            -------
            x: array_like
                Quantile values after affine transformation."""

        x = self.dist.invcdf(y)
        x = self.translate_points(x, True)
        return x / 1 # TODO ????

    def mean(self):
        """ Return the mean of the translated distribution.

            Returns
            -------
            mean: float
                The translated mean."""

        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(
            moments, self.shift, self.scale, self.center
        )
        mean = moments[0]
        return mean

    def var(self):
        """ Return the variance of the translated distribution.

            Returns
            -------
            var : float
                The translated variance."""

        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(
            moments, self.shift, self.scale, self.center
        )
        var = moments[1]
        return var

    def skew(self):
        """ Return the skewness of the translated distribution.

            Returns
            -------
            skew : float
                Skewness of the base distribution."""

        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(
            moments, self.shift, self.scale, self.center
        )
        skew = moments[2]
        return skew

    def kurt(self):
        """ Return the kurtosis of the translated distribution.

            Returns
            -------
            kurt : float
                Kurtosis of the base distribution."""

        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(
            moments, self.shift, self.scale, self.center
        )
        kurt = moments[3]
        return kurt

    def sample(self, n):
        """ Draw random samples from the translated distribution.

            Parameters
            ----------
            n : int
                Number of samples.

            Returns
            -------
            xi : ndarray
                Array of transformed samples."""

        xi = self.dist.sample(n)
        xi = self.translate_points(xi, True)
        return xi

    def moments(self):
        """ Return the first four moments of the translated distribution.

            Returns
            -------
            moments : list
                List of moments ``[mean, variance, skewness, kurtosis]``."""

        moments = self.dist.moments()
        moments = TranslatedDistribution.translate_moments(
            moments, self.shift, self.scale, self.center
        )
        return moments

    def get_base_dist(self):
        """ Return the underlying base distribution.

            Returns
            -------
            dist_germ : Distribution
                The untransformed base distribution."""

        dist_germ = self.dist.get_base_dist()
        return dist_germ

    @staticmethod
    def translate_points_forward(x, shift, scale, center):
        """ Apply the forward affine transformation:

            y = (x - center) * scale + center + shift

            Parameters
            ----------
            x : array_like
                Values from the base distribution.

            shift : float
                Translation offset.

            scale : float
                Scaling factor.

            center : float
                Center of scaling.

            Returns
            -------
            y : array_like
                Transformed points."""

        y = (x - center) * scale + center + shift
        return y

    @staticmethod
    def translate_points_backwards(x, shift, scale, center):
        """ Apply the inverse affine transformation:

            x = (y - shift - center) / scale + center

            Parameters
            ----------
            x : array_like
                Values from the translated distribution.

            shift : float
                Translation offset.

            scale : float
                Scaling factor.

            center : float
                Center of scaling.

            Returns
            -------
            y: array_like
                Points mapped back to the base distribution domain."""

        y = (x - shift - center) / scale + center
        return y

    @staticmethod
    def translate_moments(m, shift, scale, center):
        """ Transform the moments of a distribution under an affine transformation.

            Parameters
            ----------
            m : list
                Moments of the base distribution in the form
                ``[mean, variance, skewness, kurtosis]``.

            shift : float
                Translation offset.

            scale : float
                Scaling factor.

            center : float
                Center of scaling.

            Returns
            -------
            m: list
                Transformed moments."""
        
        if len(m) >= 1:
            m[0] = TranslatedDistribution.translate_points_forward(
                m[0], shift, scale, center
            )
        if len(m) >= 2:
            m[1] = m[1] * scale**2
        # Higher (standardized) moments like skewness or kurtosis are
        # not affected by neither shift nor scale
        return m
