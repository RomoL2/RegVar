# coding=future_fstrings
from __future__ import print_function

__license__ = "MIT"
__version__ = "0.9.6"
__authors__ = ["Marvin Jens"]
__email__ = "mjens@mit.edu"

import os
import logging
import numpy as np
import pickle
import shelve
import hashlib
import functools
from collections import defaultdict
cached_objects = defaultdict(dict)

def key_to_hash(key):
    return hashlib.md5(key.encode()).hexdigest()

def array_to_hash(a):
    return "array_{0}_{1}".format(a.shape, hashlib.md5(a.tobytes()).hexdigest())

def list_to_hash(ll):
    m = hashlib.sha256()
    for x in ll:
        m.update(x)

    return m.hexdigest()

def long_str_to_hash(s, thresh=20):
    s = str(s)
    if len(s) < thresh:
        return s
    else:
        m = hashlib.sha256()
        m.update(s.encode('utf-8'))
        return m.hexdigest()

def args_to_key(argc, kwargs, self, func_name):
    kw = dict(kwargs)
    kw.pop('_do_not_cache', None)
    kw.pop('_do_not_pickle', None)
    kw.pop('_do_not_unpickle', None)
    kw.pop('_argc_key', None)
    kw.pop('_kw_key', None)
    
    def to_str(x):
        if type(x) == np.ndarray:
            return array_to_hash(x)
        else:
            return str(x)

    argc_key = kwargs.get("_argc_key", "_".join([to_str(a) for a in argc]) )
    kw_key = kwargs.get("_kw_key", "__".join(["{0}={1}".format(k,to_str(v)) for k,v in sorted(kwargs.items()) ]))
    
    if self:
        key = f"{self.cache_key}.{func_name}.{argc_key}.{kw_key}"
    else:
        key = f"{func_name}.{argc_key}.{kw_key}"
    
    return key, kw


_load_from_shelf = True

def shelved(sname, depends=[]):
    def _shelved(func):
        @functools.wraps(func)
        def wrapper(self, *argc, **kwargs):
            # key = f'{self.rbp} {self.cell} detrend={self.detrend} scale={self.scale} predict_on_clip={self.predict_on_clip} expr_cutoff={self.expr_cutoff}__'
            key_parts = \
                [sname, ] + \
                [getattr(self, dep, None) for dep in depends] + \
                list(argc) + \
                ["{}={}".format(k, kwargs[k]) for k in sorted(kwargs.keys())]

            key = " ".join([long_str_to_hash(k) for k in key_parts])
            execute = True

            if key in self.shelf and _load_from_shelf:
                if kwargs.get('debug', False):
                    logging.debug(f'shelved(key={key}) loading ')
                try:
                    res = self.shelf[key]
                except Exception as E:
                    logging.error(f"caught exception {E}")
                else:
                    execute = False
                
                # if res is not None:
                #     a = res.sum()
                # else:
                #     a = None

                # print("hit", sname, gene_id, a)
                # print("got {key} from shelf: {res}".format(**locals()))
            if execute:
                res = func(self, *argc, **kwargs)
                self.shelf[key] = res
                if kwargs.get('debug', False):
                    logging.debug(f'shelved(key={key}) storing ')
            return res
        
        return wrapper
    return _shelved

def get_cache_sizes():
    cache_size = []
    for k, d in list(cached_objects.items()):
        cache_size.append( (np.array(list(d.values())).sum(), k) )

    return sorted(cache_size)[::-1]

def _dump_cache_sizes():
    for size, name in get_cache_sizes():
        print(f"{name}\t{size}")

class CachedBase(object):
    """
    Base class for anything that wants to use transparent caching and/or 
    pickling by use of the @cached or @pickled decorators. Adds the 
    minimum hooks required to make this work.
    """

    pkl_path = "./.pkl/"
    debug_caching = False # set to True to get A LOT of debug output from the caching framework
    
    # change any of the below, on instance or class level, to tune behaviour 
    # of the caching framework
    _do_not_cache = False
    _do_not_pickle = False
    _do_not_unpickle = False
    
    def __init__(self, **kwargs):
        self._cache_names = []
        self.cache_logger = logging.getLogger('cache.CachedBase')
        
        for k,v in list(kwargs.items()):
            if k.startswith('_'):
                #print "setting",k,v
                setattr(self, k, v)

        #self._do_not_cache = True # DEBUG!!

    def init_shelf(self, param_key):
        self.param_key = param_key
        self.shelf = shelve.open(
            "{self.param_key}_shelf".format(self=self),
            protocol=-1, 
            flag = 'c'
        )

    def __dump_cache_inventory(self):
        import sys
        print(f"Cache inventory of {self.cache_key}")
        for cache_name in sorted(self._cache_names):
            cache = getattr(self, cache_name)
            print("cache '{} has {} entries:".format(cache_name, len(cache)))
            for k in sorted(cache.keys()):
                v = cache[k]
                print("  '{}' : {:.2f}kb".format(k, sys.getsizeof(v) / 1024.))

    def _store(self, cache_name, key, value):
        cache = getattr(self, cache_name)        
        cache[key] = value

        global cached_objects
        import sys
        cached_objects["{}.{}".format(self.cache_key, cache_name)][key] = sys.getsizeof(value)

    def _clear(self, cache_name):
        setattr(self, cache_name, dict() )

        global cached_objects
        cached_objects.pop("{}.{}".format(self.cache_key, cache_name), None)

    @property
    def cache_key(self):
        """
        This needs to be overridden by each subclass, unless class attributes
        really do not influence the identity of the cached results.
        """
        return self.__class__.__name__
    
    def cache_preload(self, func_name, value, argc=(), kwargs={}):
        
        cache_name = "__cached_{name}".format(name=func_name)
        
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )

        key, kw = args_to_key(argc, kwargs, self, func_name)
        # getattr(self, cache_name)[key] = value
        self._store(cache_name, key, value)
        if self.debug_caching:
            self.cache_logger.debug("cache_preload {0} '{1}' to {2}".format(cache_name, key, value) )
    
    def cache_flush(self, cache_names = [], deep=False):
        if not cache_names:
            cache_names = self._cache_names
        self.cache_logger.debug("{0} flushing caches '{1}'".format(self.cache_key, cache_names) )
        for cache_name in cache_names:
            if deep:
                cache = getattr(self, cache_name, dict())
                for res in list(cache.values()):
                    if isinstance(res, CachedBase):
                        res.cache_flush()
            self._clear(cache_name)

    def cache_debug(self):
        for name in self._cache_names:
            print(">>>", self.cache_key, name)
            for k,v in sorted(getattr(self, name).items()):
                print("  '{0}' : '{1}'".format(k,v))
    
    def drop_pickle(self, func_name, *argc, **kwargs):
        func = getattr(self, func_name)
        pkl_key, kw = args_to_key(argc, kwargs, self, func.__name__)
        pkl_name = getattr(func, "pkl_name", "{pkl_hash}.pkl".format(pkl_hash = key_to_hash(pkl_key)))
        self.cache_logger.warning("dropping pickle {} for {}(argc={},kw={})".format(pkl_name, func_name, str(argc), str(kw)))
        fname = os.path.join(self.pkl_path, pkl_name)
        if os.path.exists(fname):
            self.cache_logger.warning("deleting {}".format(fname))
            os.remove(fname)
        else:
            self.cache_logger.warning("not found")

    def __del__(self):
        self.cache_flush()

def cached(func):
    """
    Decorator for class methods that keeps the results of the first call and 
    returns the cached result for subsequent calls. Works by adding a 
    "__cached_<func_name>" dictionary to the decorated method's class instance.
    """
    # TODO: 
    # * clean up into baseclass (or meta class?) of its own
    
    cache_name = "__cached_{name}".format(name=func.__name__)
    
    def cached_func(self, *argc, **kwargs):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, dict() )
            self._cache_names.append(cache_name)

        #if self.debug_caching:
            #self.cache_logger.debug("cached function {0} of {1} called with argc={2} kw={3}".format(func.__name__, self, argc, kwargs) )

        key, kw = args_to_key(argc, kwargs, self, func.__name__)
        
        cache = getattr(self, cache_name)
        if not key in cache:
            if self.debug_caching:
                self.cache_logger.debug("{0} cache-miss '{1}'".format(cache_name, key) )
                #self.cache_debug()

            if getattr(self, '_do_not_cache', False) or kwargs.get('_do_not_cache', False):
                if self.debug_caching:
                    self.cache_logger.debug("! NOT CACHING: calling {0} of {1} called with argc={2} kw={3}".format(func.__name__, self, argc, kwargs) )

                # override caching, but allow pre-loading!
                return func(self, *argc, **kw)
            else:
                #if self.debug_caching:
                    #self.cache_logger.debug("! calling {0} of {1} called with argc={2} kw={3}".format(func.__name__, self, argc, kwargs) )

                self._store(cache_name, key, func(self, *argc, **kw))
        else:
            if self.debug_caching:
                self.cache_logger.debug("{0} cache-hit '{1}'".format(cache_name, key) )
            
        return cache[key]
    
    cached_func.__name__ = func.__name__
    return cached_func
  
def monitored(func):
    
    def monitored_func(self, *argc, **kwargs):
        res = None
        new = False

        pkl_key, kw = args_to_key(argc, kwargs, self, func.__name__)

        # allow override
        pkl_name = getattr(func, "pkl_name", "{pkl_hash}.pkl".format(pkl_hash = key_to_hash(pkl_key)))
        
        # get the result from call or un-pickle
        fname = os.path.join(self.pkl_path, pkl_name)
        found = os.path.exists(fname)

        def shortstr(x):
            s = str(x)
            if len(s) > 8:
                s = s[:8] + "_" + hashlib.md5(s).hexdigest()[:6]
            return s

        argc_short = ", ".join([shortstr(a) for a in argc])
        kw_short = ", ".join(["{k}={s}".format(k=k, s=shortstr(v)) for k,v in sorted(kwargs.items())])
        short_self = shortstr(self.cache_key)
        self.cache_logger.warning('monitored call to {func.__name__} with self.cache_key={short_self} argc={argc_short} kw={kw_short} -> {pkl_name} pickle exists={found}'.format(**locals()))
        return func(self, *argc, **kwargs)

    monitored_func.__name__ = func.__name__
    return monitored_func

def pickled(func):
    """
    Decorator for class methods that returns an un-pickled result if it exists. 
    Otherwise, stores the result of the call in a pickle file. Requires that the 
    class has an out_path attribute and a pickle_key method that returns a distinct 
    key for all the parameters that influence the results, ensuring that the correct
    object is unpickled.
    """
    
    def pickled_func(self, *argc, **kwargs):
        res = None
        new = False

        pkl_key, kw = args_to_key(argc, kwargs, self, func.__name__)

        # allow override
        pkl_name = getattr(func, "pkl_name", "{pkl_hash}.pkl".format(pkl_hash = key_to_hash(pkl_key)))
        
        # get the result from call or un-pickle
        fname = os.path.join(self.pkl_path, pkl_name)
        if getattr(self, '_do_not_unpickle', False) or kwargs.get('_do_not_unpickle', False):
            res = func(self, *argc, **kw)
            new = True
        
        elif os.path.exists(fname):
            self.cache_logger.debug("un-pickling '{0}' as '{1}'".format(pkl_key, pkl_name) )
            res = pickle.load(open(fname,'rb'), encoding='bytes')
            new = False
            
        else:
            res = func(self, *argc, **kw)
            new = True

        # store the result, if new and not disabled
        if new and (not (getattr(self, '_do_not_pickle', False) or kwargs.get('_do_not_pickle', False))):
            self.cache_logger.debug("storing pickle of '{0}' as '{1}'".format(pkl_key, pkl_name) )
            try:
                os.makedirs(self.pkl_path)
            except OSError:
                # already exists
                pass
            pickle.dump(res, open(fname,'wb'), protocol=-1)
        
        return res
    
    pickled_func.__name__ = func.__name__
    return pickled_func

if __name__ == "__main__":

    class A(CachedBase):
        @cached
        def get_data(self, N):
            return np.zeros(N)
    
    class B(CachedBase):
        @cached
        def get_nested(self, N):

            res = []
            for i in range(N):
                a = A()
                b = a.get_data(100)
                res.append(a)

            return res
    
    a0 = A()
    a0.get_data(20000)

    b = B()
    b.get_nested(20)

    for size, name in get_cache_sizes():
        print(size, name)
                

