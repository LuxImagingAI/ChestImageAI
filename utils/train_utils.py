import os, glob, json, sys
import copy
import collections
import torchvision

def instantiate_object(class_name, param_dict, module='__main__', **kwargs):
    ''' creates object of type `class_name` with keyword arguments
    
        class_name might contain module prefix (e.g. `torch.data.Dataset`)
        or module might be specified seperatly (e.g. `class_name = Dataset, module = torch.data` )
        `param_dict` and `kwargs` are both used as keyword arguments for the object
    '''
    
    module_hierachy = class_name.split('.')
    module_or_class = sys.modules[module]
    for sub_name in module_hierachy:
        module_or_class = getattr(module_or_class, sub_name)
    obj_ = module_or_class(**param_dict, **kwargs)
    return obj_



class EndlessIterator():
    ''' Wrapper for dataloaders: instead of stop iteration, creates new iter'''
    
    
    def __init__(self, dataloader):
        
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.epochs = 0
    
    def __next__(self):
        return self.next_batch()
    
    def next_batch(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.epochs += 1
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        return batch
    

def transform_pipe_factory(transform_listdict, keys, obj='__main__', pop_key=[]):
    """ creates transform pipeline from all keys in dict"""    
    for k in keys:
        transform_objects = [transform_factory(k2, v, obj, pop_key) for k2, v in transform_listdict[k]]
    return torchvision.transforms.Compose(transform_objects)
    
    
def transform_factory(name, param_dict={}, obj='__main__', pop_key=[]):
    ''' creates transform object from name and parameter dict 
    
        name: classname, might be precceded by module hierachy (i.e. torchvision.transforms.Resize) 
        param_dict:         
    '''
    
    # create transform obj
    module_hierachy = name.split('.')
    obj = sys.modules[obj]
    for sub_name in module_hierachy:
        obj = getattr(obj, sub_name)
    
    # remove pop_key from params, i.e. from keys and 
    # from all parameter lists
    param = copy.deepcopy(param_dict)
    for k in pop_key:
        if k in param['keys']:
            ix = param['keys'].index(k)
            for k, v in param.items():
                if type(v) == list:
                    _ = v.pop(ix)
    transform = obj(**param)
    
    return transform