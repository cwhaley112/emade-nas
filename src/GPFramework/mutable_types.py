from sqlalchemy.ext.mutable import Mutable

def simple_type(obj):
    return type(obj) in [type(None), int, float, complex, bool, str]

def mutable_type(obj):
    return simple_type(obj) or isinstance(obj, MutableType)


class MutableType(Mutable):
    """Base class for mutable record types which propagate change events to parents"""

    def register_parent(self, parent):
        if not isinstance(parent, MutableType):
            raise ValueError('Parent must be a MutableType')
        if not hasattr(self, '_mutable_parents'):
            setattr(self, '_mutable_parents', [])
        getattr(self, '_mutable_parents').append(parent)

    def unregister_parent(self, parent):
        if hasattr(self, '_mutable_parents'):
            getattr(self, '_mutable_parents').remove(parent)

    def changed(self):
        if hasattr(self, '_mutable_parents'):
            parents = getattr(self, '_mutable_parents')
            if len(parents) > 0:
                for parent in parents:
                    parent.changed()
                return
        Mutable.changed(self)


class MutableDict(MutableType, dict):
    def __init__(self, d:dict={}):
        for key in d:
            value = d[key]
            if not simple_type(key):
                raise ValueError('Key is not a simple type')
            if not mutable_type(value):
                raise ValueError('Value is not a mutable type')
        dict.__init__(self, d)
        for key in d:
            value = d[key]
            if isinstance(value, MutableType):
                value.register_parent(self)

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableDict):
            if isinstance(value, dict):
                return MutableDict(value)
            return Mutable.coerce(key, value)
        else:
            return value

    def __setitem__(self, key, value):
        if not simple_type(key):
            raise ValueError('Key is not a simple type')
        if not mutable_type(value):
            raise ValueError('Value is not a mutable type')
        dict.__setitem__(self, key, value)
        if isinstance(value, MutableType):
            value.register_parent(self)
            self.changed()

    def __delitem__(self, key):
        val = self[key]
        if isinstance(val, MutableType):
            val.unregister_parent()
        dict.__delitem__(self, key)
        self.changed()

    def popitem(self):
        val = dict.popitem(self)
        if isinstance(val, MutableType):
            val.unregister_parent()
        self.changed()
        return val

    def pop(self, key):
        val = dict.popitem(self, key)
        if isinstance(val, MutableType):
            val.unregister_parent()
        self.changed()
        return val

    def update(self, d):
        for k in d:
            value = d[k]
            if not mutable_type(value):
                raise ValueError('Value is not a mutable type')
        dict.update(self, d)
        for k in d:
            value = d[k]
            if isinstance(value, MutableType):
                value.register_parent(self)
        self.changed()

    def clear(self):
        for key in self:
            value = d[key]
            if isinstance(value, MutableType):
                value.unregister_parent(self)
        dict.clear(self)
        self.changed()

    def __getstate__(self):
        state = {k: self[k].__getstate__() if isinstance(self[k], MutableType) else self[k] for k in self}
        state['__type__'] = 'd'
        return state

    def __setstate__(self, state):
        self.clear()
        if not isinstance(state, dict):
            raise ValueError('Invalid state')
        for k in state:
            if k == '__type__':
                continue
            val = state[k]
            if simple_type(val):
                self[k] = val
            elif isinstance(state, dict):
                try:
                    ind = ['d', 'l', 's', 'o'].index(val['__type__'])
                except ValueError as e:
                    raise ValueError('Invalid state')
                typ = [MutableDict, MutableList, MutableSet, MutableObject][ind]
                val_obj = typ()
                val_obj.__setstate__(val)
                self[k] = val_obj
                val_obj.register_parent(self)
            else:
                raise ValueError('Invalid state')


class MutableList(MutableType, list):
    def __init__(self, l:list=[]):
        for x in l:
            if not mutable_type(x):
                raise ValueError('Value {} is not a mutable type. {}'.format(x, type(x)))
        list.__init__(self, l)
        for x in l:
            if isinstance(x, MutableType):
                x.register_parent(self)

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableList):
            if isinstance(value, list):
                return MutableList(value)
            return Mutable.coerce(key, value)
        else:
            return value

    def __setitem__(self, key, value):
        if not mutable_type(value):
            raise ValueError('Value is not a mutable type')
        list.__setitem__(self, key, value)
        if isinstance(value, MutableType):
            value.register_parent(self)
        self.changed()

    def __delitem__(self, key):
        val = self[key]
        dict.__delitem__(self, key)
        if isinstance(value, MutableType):
            value.unregister_parent(self)
        self.changed()

    def append(self, value):
        if not mutable_type(value):
            raise ValueError('Value is not a mutable type')
        list.append(self, value)
        if isinstance(value, MutableType):
            value.register_parent(self)
        self.changed()

    def extend(self, values):
        for value in values:
            if not mutable_type(value):
                raise ValueError('Value is not a mutable type')
        list.extend(self, values)
        for value in values:
            if isinstance(value, MutableType):
                value.register_parent(self)

    def clear(self):
        for value in self:
            if isinstance(value, MutableType):
                value.unregister_parent(self)
        list.clear(self)
        self.changed()

    def __getstate__(self):
        state = {i: x.__getstate__() if isinstance(x, MutableType) else x for i, x in enumerate(self)}
        state['__type__'] = 'l'
        return state

    def __setstate__(self, state):
        self.clear()
        if not isinstance(state, dict):
            raise ValueError('Invalid state')
        for k in state:
            if k == '__type__':
                continue
            val = state[k]
            if simple_type(val):
                self.append(val)
            elif isinstance(state, dict):
                try:
                    ind = ['d', 'l', 's', 'o'].index(val['__type__'])
                except ValueError as e:
                    raise ValueError('Invalid state')
                typ = [MutableDict, MutableList, MutableSet, MutableObject][ind]
                val_obj = typ()
                val_obj.__setstate__(val)
                self.append(val_obj)
                val_obj.register_parent(self)
            else:
                raise ValueError('Invalid state')


class MutableSet(MutableType, set):
    def __init__(self, s:set={}):
        for x in l:
            if not mutable_type(x):
                raise ValueError('Value is not a mutable type')
        set.__init__(self, s)
        for x in l:
            if isinstance(x, MutableType):
                x.register_parent(self)

    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableSet):
            if isinstance(value, set):
                return MutableSet(set)
            return Mutable.coerce(key, value)
        else:
            return value

    def add(self, value):
        if not mutable_type(value):
            raise ValueError('Value is not a mutable type')
        set.add(self, value)
        if isinstance(value, MutableType):
            value.register_parent(self)
        self.changed()

    def remove(self, value):
        set.remove(self, value)
        if isinstance(value, MutableType):
            value.unregister_parent(self)
        self.changed()

    def discard(self, value):
        val = set.discard(self, value)
        if isinstance(val, MutableType):
            val.unregister_parent(self)
        self.changed()
        return val

    def pop(self):
        val = set.pop(self)
        if isinstance(val, MutableType):
            val.unregister_parent(self)
        self.changed()
        return val

    def clear(self):
        def clear(self):
            for value in self:
                if isinstance(value, MutableType):
                    value.unregister_parent(self)
            list.clear(self)
        self.changed()

    def __getstate__(self):
        state = {i: x.__getstate__() if isinstance(x, MutableType) else x for i, x in enumerate(self)}
        state['__type__'] = 's'
        return state

    def __setstate__(self, state):
        self.clear()
        if not isinstance(state, dict):
            raise ValueError('Invalid state')
        for k in state:
            if k == '__type__':
                continue
            val = state[k]
            if simple_type(val):
                self.add(val)
            elif isinstance(state, dict):
                try:
                    ind = ['d', 'l', 's', 'o'].index(val['__type__'])
                except ValueError as e:
                    raise ValueError('Invalid state')
                typ = [MutableDict, MutableList, MutableSet, MutableObject][ind]
                val_obj = typ()
                val_obj.__setstate__(val)
                self.add(val_obj)
                val_obj.register_parent(self)
            else:
                raise ValueError('Invalid state')


class MutableObject(MutableDict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return super.__getattr__(self, key)

    def __setattr__(self, key, value):
        self[key] = value

    def __getstate__(self):
        state = MutableDict.__getstate__(self)
        state['__type__'] = 'o'
        return state
