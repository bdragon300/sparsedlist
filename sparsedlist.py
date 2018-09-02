import itertools
from pyskiplist import SkipList
from collections import MutableSequence, Iterable


DOES_NOT_EXIST = object()


class SparsedList(MutableSequence):
    def __init__(self, initlist=None, inititems=None):
        self.data = SkipList()

        if initlist is not None:
            for i, v in enumerate(initlist):
                self.data.insert(i, v)

        if inititems is not None:
            for i, v in inititems:
                self.data.insert(i, v)

    def __repr__(self): return 'SparsedList{' + str(dict(self.data.items())) + '}'

    def __eq__(self, other):
        return len(self.data) == len(self.__cast(other)) \
               and all(a[1] == b[1] and a[0] == b[0] for a, b in zip(self.data, self.__cast(other)))

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def __cast(other):
        return other.data if isinstance(other, SparsedList) else other

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __len__(self):
        try:
            return self.tail() + 1
        except IndexError:
            return 0

    def __getitem__(self, item):
        def objs(start, stop, step):
            c = start or 0
            step = step or 1

            items = self.data.items(start=start, stop=stop)
            for i in items:
                if i[0] == c:
                    yield i[1]
                    c += step
                elif i[0] > c:
                    raise IndexError("Item with index '{}' does not exist".format(c))

            if c == (start or 0):
                raise IndexError("Item with index '{}' does not exist".format(c))

        if isinstance(item, slice):
            start, stop, step = self._slice_indexes(item)

            return objs(start, stop, step)
        else:
            item = int(item)
            if item < 0:
                last_ind = self.data[-1][0]  # IndexError if empty
                item = last_ind + item + 1
                if item < 0:
                    raise IndexError("Negative index overlaps the list start")

            val = self.data.search(item, default=DOES_NOT_EXIST)
            if val is DOES_NOT_EXIST:
                raise IndexError("Item with index '{}' does not exist".format(item))

            return val

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError('Can only assign an iterable')

            start, stop, step = self._slice_indexes(key)
            step = step or 1
            start = start or 0
            c = start
            vi = iter(value)

            while stop is None or c < stop:
                try:
                    self.data.replace(c, next(vi))
                except StopIteration:
                    # Remove the rest elements in slice if it longer than given iterable
                    for i in self.data.items(start=c, stop=stop):
                        if (i[0] - start) % step:  # Dont touch items which does not fall into steps
                            continue

                        self.data.remove(i[0])
                    return

                c += step

        else:
            key = int(key)
            if key < 0:
                last_ind = self.data[-1][0]  # IndexError if empty
                key = last_ind + key + 1
                if key < 0:
                    raise IndexError("Negative index overlaps the list start")

            self.data.replace(key, value)

    def __delitem__(self, key):
        try:
            key = int(key)
            if key < 0:
                last_ind = self.data[-1][0]  # IndexError if empty
                key = last_ind + key + 1
                if key < 0:
                    raise IndexError("Negative index overlaps the list start")

            self.data.remove(key)
        except KeyError:
            raise IndexError("Item '{}' does not exist".format(key))

    def __iter__(self):
        return iter(self.data.values())

    def __reversed__(self):
        raise TypeError("'SparsedList' object is not reversible")

    def __add__(self, other):
        obj = self.__class__()
        try:
            offset = self.tail() + 1
        except IndexError:
            offset = 0

        if isinstance(other, SparsedList):
            other = ((i + offset, v) for i, v in other.data)
        else:
            other = enumerate(other, start=offset)

        for i in itertools.chain(self.data, other):
            obj.data.insert(*i)

        return obj

    def __radd__(self, other):
        obj = self.__class__()

        if isinstance(other, SparsedList):
            try:
                offset = other.tail() + 1
            except IndexError:
                offset = 0
            other = other.data
        else:
            offset = len(other)
            other = enumerate(other)

        this = ((i + offset, v) for i, v in self.data)

        for i in itertools.chain(other, this):
            obj.data.insert(*i)

        return obj

    def __iadd__(self, other):
        try:
            offset = self.tail() + 1
        except IndexError:
            offset = 0

        if isinstance(other, SparsedList):
            other = ((i + offset, v) for i, v in other.data)
        else:
            other = enumerate(other, start=offset)

        for i in other:
            self.data.insert(*i)

        return self

    def __mul__(self, n):
        obj = self.__class__()

        try:
            offset = self.tail() + 1
        except IndexError:
            offset = 0

        for c in range(0, offset * n, offset):
            for i, v in self.data:
                obj.data.insert(i + c, v)

        return obj

    __rmul__ = __mul__

    def __imul__(self, n):
        try:
            offset = self.tail() + 1
        except IndexError:
            offset = 0

        for c in range(offset, offset * n, offset):
            for i, v in self.data.items(stop=offset):
                self.data.insert(i + c, v)

        return self

    def __or__(self, other):
        raise NotImplementedError

    def __ror__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError

    def __and__(self, other):
        raise NotImplementedError

    def __rand__(self, other):
        raise NotImplementedError

    def __iand__(self, other):
        raise NotImplementedError

    def __xor__(self, other):
        raise NotImplementedError

    def __rxor__(self, other):
        raise NotImplementedError

    def __ixor__(self, other):
        raise NotImplementedError

    def __rshift__(self, other):
        raise NotImplementedError

    def __rrshift__(self, other):
        raise NotImplementedError

    def __irshift__(self, other):
        raise NotImplementedError

    def __lshift__(self, other):
        raise NotImplementedError

    def __rlshift__(self, other):
        raise NotImplementedError

    def __ilshift__(self, other):
        raise NotImplementedError

    def insert(self, index, value):
        raise NotImplementedError

    def append(self, value):
        """Append given value in place after the last item"""
        self.data.insert(self.tail() + 1, value)

    def extend(self, items):
        """
        Extend (merge) SparsedList with given items. Already existing items will be overwritten
        :param items: key/value pairs iterable
        """
        for i, v in items:
            self.data.replace(i, v)

    def clear(self):
        """Clear all data"""
        self.data.clear()

    def reverse(self):
        raise TypeError("'SparsedList' object is not reversible")

    def pop(self, index=-1):
        """Pop the item with given index. Negative indexes counted from position of the last existing item"""
        if index < 0:
            index = max(self.tail() + index + 1, 0)

        try:
            return self.data.pop(index)
        except KeyError:
            raise IndexError('Pop from empty SparsedList')

    def remove(self, value):
        """Remove the first item from the list whose value is equal to x. ValueError is raised if value not found"""
        ind = self.index(value)
        self.data.remove(ind)

    def sort(self, *args, **kwds):
        raise TypeError("'SparsedList' object cannot be sorted")

    def copy(self):
        obj = self.__class__()
        for p in self.data.items():
            obj.data.insert(*p)

        return obj

    def index(self, value, start=None, stop=None):
        """
        Return zero-based index in the list of the first item whose value is equal to x.
        Raises a ValueError if there is no such item.
        """
        if start is not None and start < 0:
            start = max(self.tail() + start + 1, 0)
        if stop is not None and stop < 0:
            stop = max(self.tail() + stop + 1, 0)

        for i, v in self.data.items(start, stop):
            if v == value:
                return i

        raise ValueError("'{}' is not in SparsedList".format(value))

    def count(self, *args, **kwargs):
        """
        Return count of existing elements
        If `item` keyword parameter given then method returns how many times `item` occurs in list
        :param item: Optional. Value to search for in Sparsed list
        """
        if args or kwargs:
            item = (args and args[0]) or (kwargs and kwargs.get('item'))
            return len([x for x in self.data.values() if x == item])
        else:
            return len(self.data)

    def items(self, start=None, stop=None):
        if start is not None and start < 0:
            start = max(self.tail() + start + 1, 0)
        if stop is not None and stop < 0:
            stop = max(self.tail() + stop + 1, 0)

        return self.data.items(start=start, stop=stop)

    def tail(self):
        """Return index of the last element"""
        return self.data[-1][0]

    @staticmethod
    def _merge(a, b):
        """
        Returns new instance merged from `a` and `b` iterables. They must have ((index, value),...) format.
        Values from `a` will be overwritten by `b` on the same position.
        :param a: iterable
        :param b: iterable
        :return: SkipList object
        """
        obj = SkipList()
        aindexes = set(a[0] for a in a)
        bindexes = set(b[0] for b in b)
        av = (-1, -1)
        bv = (-1, -1)
        ai, bi = iter(a), iter(b)

        for i in sorted(aindexes | bindexes):
            try:
                if av is not None:
                    if av[0] < i:
                        av = next(ai)
                    if av[0] == i:
                        obj.replace(*av)
            except StopIteration:
                av = None
            try:
                if bv is not None:
                    if bv[0] < i:
                        bv = next(bi)
                    if bv[0] == i:
                        obj.replace(*bv)
            except StopIteration:
                bv = None

        return obj

    def _slice_indexes(self, s):
        """
        Calculate positive index bounds from slice. If slice param is None, then it will be left as None
        :param s: slice object
        :return: start, stop, step
        """
        start, stop, step = s.start, s.stop, s.step

        try:
            if start is not None:
                start = int(start)
                if start < 0:
                    last_ind = self.data[-1][0]  # IndexError if empty
                    start = max(last_ind + start + 1, 0)

            if stop is not None:
                stop = int(stop)
                if stop < 0:
                    last_ind = self.data[-1][0]  # IndexError if empty
                    stop = max(last_ind + stop + 1, 0)

        except IndexError:
            raise IndexError('Slice is out of range')

        if step is not None:
            if step < 0:
                raise ValueError('Negative slice step is not supported')
            elif step == 0:
                raise ValueError('Slice step cannot be zero')

        return start, stop, step


__all__ = ('SparsedList', )
