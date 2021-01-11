"""Experiment Parameter Managment

Example:

Attributes:

Todo:

"""


import collections

def deep_update(source, overrides, warn_on_overwrites=False):
    """
    Update values in a !nested! mapping (dictionary)
    
    Args:
        source (mapping):    gets updatet in place
        overrides (mapping): keys and values to update
    
    Returns:
        mapping: updated source
    
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            if warn_on_overwrites and (source.get(key, None) != overrides[key]):
                print(f'{key}: {source.get(key, "")} -> {overrides.get(key, "")} by injection')
            source[key] = overrides[key]

    return source


class ParameterStore():
    ''' Stores '''
    
    def __init__(self, defaults = {}, overwrites = {}):
        self.params = defaults
        self.overwrites = overwrites
    
    def __setitem__(self, name, value):
        subdict = None
        name_split = name.split('::')
        for name in name_split:
            subdict = subdict or self.params
            subdict = subdict.setdefault(name, {})
        deep_update(subdict, value)
        deep_update(subdict, self.overwrites.get(name, {}), True)           
            
    def __getitem__(self, name):
        return self.params[name]
    
    def __getattr__(self, name):
        if name in ['params', 'overwrites']:
            return super().__getattr__(name)
        else:
            return self.params[name]
    
    def __setattr__(self, name, value):
        if name in ['params', 'overwrites']:
            super().__setattr__(name, value)
        else:
            self[name] = value
