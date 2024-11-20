from __future__ import print_function
import shelve
import json_tricks as json
import os, sys

spath = sys.argv[1]

def fb_encode(obj, is_changed=False):
    return "FALLBACK " + str(obj)

import numpy
dtypes = {
    numpy.dtype : str,
    numpy.float32 : float,
    numpy.float64 : float,
    numpy.int8 : int,
    numpy.int32 : int,
    numpy.int64 : int,
}

def np_type_encode(obj, **kw):
    try:
        if obj in dtypes:
            return str(obj)
    except TypeError:
        pass

    return obj

def np_single_encode(obj, **kw):
    try:
        if obj.__class__ in dtypes:
            return dtypes[obj.__class__](obj)
    except TypeError:
        pass
        # if isinstance(obj, t) or obj.__class__.__name__ == t:
        #     return (obj)
    
    # print("still here???", obj, obj.__class__, obj.__class__.__name__)
    return obj
shelf = shelve.open(spath, "r")

data = {}
# rbp_conc = shelf['rbp_conc']
# for r in rbp_conc:
#     s = json.dumps(r, extra_obj_encoders=(np_single_encode, ))
#     print(s)

# s = json.dumps(shelf[k], extra_obj_encoders=(np_type_encode, ))
# print(s)
# x = shelf['gcuGCAuGcau_opt_profile_10_-1']
# print(x)
# print(json.dumps(x, extra_obj_encoders=(np_type_encode, np_single_encode)))
drop = ["indices_threshold",]

for k in sorted(shelf.keys()):
    for d in drop:
        if k.endswith(d):
            continue
    
    data[k] = shelf[k]
    # s = json.dumps(shelf[k], extra_obj_encoders=(np_type_encode, np_single_encode))


path = spath + '.json'
jstr = json.dumps(data, extra_obj_encoders = (np_type_encode, np_single_encode))
open(path, 'w').write(jstr.decode('unicode_escape'))
