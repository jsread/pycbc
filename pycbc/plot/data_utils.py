#! /usr/bin/env python

import os, sys
import math
import numpy
import operator
import lal

#############################################
#
#   Data Storage and Slicing tools
#
#############################################

class Template(object):
    """
    Class to store information about a template for plotting.
    """
    # we'll group the various parameters by type
    _intrinsic_params = [
        'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z', 'eccentricity',
        'lambda1', 'lambda2'
        ]
    _extrinsic_params = [
        'phi0', 'inclination', 'distance'
        ]
    _waveform_params = [
        'sample_rate', 'segment_length', 'duration',
        'f_min', 'f_ref', 'f_max',
        'axis_choice', 'modes_flag',
        'amp_order', 'phase_order', 'spin_order', 'tidal_order',
        'approximant', 'taper'
        ]
    _ifo_params = [
        'ifo', 'sigma'
        ]
    _tmplt_weight_params = ['weight', 'weight_function']
    _id_name = 'tmplt_id'
    __slots__ = [_id_name] + _intrinsic_params + _extrinsic_params + \
        _waveform_params + _ifo_params + _tmplt_weight_params

    def __init__(self, **kwargs):
        default = None
        [setattr(self, param, kwargs.pop(param, default)) for param in \
            self.__slots__]
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))

    # some other derived parameters
    @property
    def mtotal(self):
        return self.mass1 + self.mass2

    @property
    def mtotal_s(self):
        return lal.MTSUN_SI*self.mtotal

    @property
    def q(self):
        return self.mass1 / self.mass2

    @property
    def eta(self):
        return self.mass1*self.mass2 / self.mtotal**2.

    @property
    def mchirp(self):
        return self.eta**(3./5)*self.mtotal

    @property
    def chi(self):
        return (self.mass1*self.spin1z + self.mass2*self.spin2z) / self.mtotal

    # some short cuts
    @property
    def m1(self):
        return self.mass1

    @property
    def m2(self):
        return self.mass2

    def tau0(self, f0=None):
        """
        Returns tau0. If f0 is not specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (5./(256 * numpy.pi * f0 * self.eta)) * \
            (numpy.pi * self.mtotal_s * f0)**(-5./3.)
   
    def v0(self, f0=None):
        """
        Returns the velocity at f0, as a fraction of c. If f0 is not
        specified, uses self.f_min.
        """
        if f0 is None:
            f0 = self.f_min
        return (2*numpy.pi* f0 * self.mtotal_s)**(1./3)

    @property
    def s1(self):
        return numpy.array([self.spin1x, self.spin1y, self.spin1z])

    @property
    def s1x(self):
        return self.spin1x

    @property
    def s1y(self):
        return self.spin1y

    @property
    def s1z(self):
        return self.spin1z

    @property
    def s2(self):
        return numpy.array([self.spin2x, self.spin2y, self.spin2z])

    @property
    def s2x(self):
        return self.spin2x

    @property
    def s2y(self):
        return self.spin2y

    @property
    def s2z(self):
        return self.spin2z

    @property
    def apprx(self):
        return self.approximant


class SnglIFOInjectionParams(object):
    """
    Class that stores information about an injeciton in a specific detector.
    """
    __slots__ = ['ifo', 'end_time', 'end_time_ns', 'sigma']
    def __init__(self, **kwargs):
        [setattr(self, param, kwargs.pop(param, None)) for param in \
            self.__slots__]
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))



class Injection(Template):
    """
    Class to store information about an injection for plotting. Inherits from
    Template, and adds slots for sky location.
    """
    # add information about location and time in geocentric coordinates, and
    # the distribution from which the injections were drawn
    _inj_params = [
        'geocent_end_time', 'geocent_end_time_ns', 'ra', 'dec', 'astro_prior',
        'min_vol', 'vol_weight', 'mass_distr', 'spin_distr'
        ]
    # we'll override some of the parameters in TemplateResult
    _id_name = 'simulation_id'
    # sngl_ifos will be a dictionary pointing to instances of
    # SnglIFOInjectionParams
    _ifo_params = ['sngl_ifos']
    __slots__ = [_id_name] + Template._intrinsic_params + \
        Template._extrinsic_params + Template._waveform_params + \
        _ifo_params + _inj_params
    def __init__(self, **kwargs):
        # ensure sngl_ifos is a dictionary
        self.sngl_ifos = kwargs.pop('sngl_ifos', {})
        # set the default for the rest to None
        [setattr(self, param, kwargs.pop(param, None)) for param in \
            self.__slots__ if param != 'sngl_ifos']
        # if anything left, raise an error, as it is an unrecognized argument
        if kwargs != {}:
            raise ValueError("unrecognized arguments: %s" %(kwargs.keys()))


class Result(object):
    """
    Class to store a template with an injection, along with event information
    (ranking stat, etc.) for purposes of plotting.

    Information about the template and the injection (masses, spins, etc.)
    are stored as instances of Template and Injection; since these contain
    static slots, this saves on memory if not all information is needed.

    Information about a trigger (snr, chisq, etc.) are stored as attributes
    of the class. Since the class has no __slots__, additional statistics
    may be added on the fly.

    The class can also be used to only store information about a trigger;
    i.e., adding an injection is not necessary.
    
    To make accessing intrinsic parameters easier, set the psuedoattr_class;
    this will make the __slots__ of either the injection or the template
    attributes of this class. See set_psuedoattr_class for details. 
    """
    _psuedoattr_class = None

    def __init__(self, unique_id=None, database=None, event_id=None,
            tmplt=None, injection=None):
        self.unique_id = None
        self.database = None
        self.event_id = None
        if tmplt is None:
            self.template = Template()
        else:
            self.template = tmplt
        if injection is None:
            self.injection = Injection()
        else:
            self.injection = injection
        self._psuedoattr_class = None
        self.snr = None
        self.chisq = None
        self.chisq_dof = None
        self.new_snr = None
        self.false_alarm_rate = None
        self.uncombined_far = None
        self.false_alarm_probability = None
        # experiment parameters
        self.instruments_on = None
        self.livetime = None
        # banksim parameters
        self.effectualness = None
        self.snr_std = None
        self.chisq_std = None
        self.new_snr_std = None
        self.num_samples = None

    def __getattr__(self, name):
        """
        This will get called if __getattribute__ fails. Thus, we can use
        this to access attributes of the psuedoattr_class, if it is set.
        """
        try:
            return object.__getattribute__(self,
                '_psuedoattr_class').__getattribute__(name)
        except AttributeError:
            raise AttributeError("'Result' object has no attribute '%s'" %(
                name))

    def __setattr__(self, name, value):
        """
        First tries to set the attribute in self. If name is not in self's
        dict, next tries to set the attribute in self._psuedoattr_class.
        If that fails with an AttributeError, it then adds the name to self's
        namespace with the associated value.
        """
        try:
            object.__getattribute__(self,
                '_psuedoattr_class').__setattr__(name, value)
        except AttributeError:
            object.__setattr__(self, name, value)

    @property
    def psuedoattr_class(self):
        return self._psuedoattr_class

    def set_psuedoattr_class(self, psuedo_class):
        """
        Makes the __slots__ of the given class visible to self's namespace.
        An error is raised if self and psuedo_class have any attributes
        that have the same name, as this can lead to unexpected behavior.

        Parameters
        ----------
        psuedo_class: {self.template|self.injection}
            An instance of a class to make visible. Should be either self's
            injection or template (but can be any instance of any class).
        """
        # check that there is no overlap
        attribute_overlap = [name for name in self.__dict__ \
            if name in psuedo_class.__slots__]
        if attribute_overlap != []:
            raise AttributeError(
                "attributes %s " %(', '.join(attribute_overlap)) + \
                "are common to self and the given psuedo_class. Delete " +\
                "these attributes from self if you wish to use the given " +\
                "psuedo_class.")
        self._psuedoattr_class = psuedo_class
        
    @property
    def optimal_snr(self):
        """
        Returns the quadrature sum of the inj_sigmas divided by the distance.
        """
        return numpy.sqrt((numpy.array([ifo.sigma \
            for ifo in self.injection.sngl_ifos])**2.).sum()) / \
            self.injection.distance

    # some short cuts
    @property
    def tmplt(self):
        return self.template

    @property
    def inj(self):
        return self.injection


# FIXME: dataUtils in pylal should be moved to pycbc, and the get_val in there
# used instead
def get_arg(row, arg):
    """
    Retrieves an arbitrary argument from the given row object. For speed, the
    argument will first try to be retrieved using getattr. If this fails (this
    can happen if a function of several attributes of row are requested),
    then Python's eval command is used to retrieve the argument. The argument
    can be any attribute of row, or functions of attributes of the row
    (assuming the relevant attributes are floats or ints). Allowed functions
    are anything in Python's math library. No other functions (including
    Python builtins) are allowed.

    Parameters
    ----------
    row: any instance of a Python class
        The object from which to apply the given argument to.
    arg: string
        The argument to apply.

    Returns
    -------
    value: unknown type
        The result of evaluating arg on row. The type of the returned value
        is whatever the type of the data element being retreived is.
    """
    try:
        return getattr(row, arg)
    except AttributeError:
        row_dict = dict([ [name, getattr(row,name)] for name in dir(row)])
        safe_dict = dict([ [name,val] for name,val in \
            row_dict.items()+math.__dict__.items() \
            if not name.startswith('__')])
        return eval(arg, {"__builtins__":None}, safe_dict)


def result_in_range(result, test_dict):
    cutvals = [(get_arg(result, criteria), low, high) \
        for criteria,(low,high) in test_dict.items()] 
    return not any(x < low or x >= high for (x, low, high) in cutvals)


def result_is_match(result, test_dict):
    try:
        matchvals = [(getattr(result, criteria), targetval) 
            for criteria,targetval in test_dict.items()]
    except:
        matchvals = [(get_arg(result, criteria), targetval) \
            for criteria,targetval in test_dict.items()]
    return not any(x != targetval for (x, tagetval) in matchvals)


def apply_cut(results, test_dict):
    return [x for x in results if result_in_range(x, test_dict)]

def slice_results(results, test_dict):
    return apply_cut(results, test_dict)


def parse_results_cache(cache_file):
    filenames = []
    f = open(cache_file, 'r')
    for line in f:
        thisfile = line.split('\n')[0]
        if os.path.exists(thisfile):
            filenames.append(thisfile)
    f.close()
    return filenames

