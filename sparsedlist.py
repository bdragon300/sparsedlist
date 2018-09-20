import itertools
from pyskiplist import SkipList
from collections import MutableSequence, Iterable
from copy import deepcopy


DOES_NOT_EXIST = object()


class SparsedList(MutableSequence):
    def __init__(self, initlist=None, inititems=None, required=False):
        """
        :param initlist: Optional. Initial data. Elements will be placed sequentallu
        :param inititems: Optional. Initial items pairs.
        :param required: Optional. If True, getting unset elements causes IndexError. Otherwise, unset elements will
            be substituted by None. Default is False.
        """
        self.data = SkipList()
        self._required = required

        if initlist is not None:
            for i, v in enumerate(initlist):
                self.data.insert(i, v)

        if inititems is not None:
            for i, v in inititems:
                self.data.insert(i, v)

    def _clone(self):
        return self.__class__(required=self._required)

    def _unset(self, index):
        if not self._required:
            return None
        else:
            raise IndexError("Item with index '{}' does not exist".format(index))

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
        return any(x == item for x in self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        def objs(start, stop, step):
            c = start or 0
            step = step or 1

            # E.g. [5:5] or [10:5]
            if stop is not None and (start or 0) >= stop:
                return []

            items = self.data.items(start=start, stop=stop)  # generator
            for i in items:
                while c < i[0]:
                    yield self._unset(c)
                    c += step

                if c == i[0]:
                    yield i[1]
                    c += step

            while stop is None or c < stop:
                yield self._unset(c)
                c += step

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
                return self._unset(item)

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
        if isinstance(key, slice):
            start, stop, step = self._slice_indexes(key)

            step = step or 1
            start = start or 0
            stop = stop or self.tail() + 1
            c = start
            while stop is None or c < stop:
                try:
                    self.data.remove(c)
                except KeyError:
                    pass

                c += step
        else:
            try:
                key = int(key)
                if key < 0:
                    last_ind = self.data[-1][0]  # IndexError if empty
                    key = last_ind + key + 1
                    if key < 0:
                        raise IndexError("Negative index overlaps the list start")

                self.data.remove(key)
            except KeyError:
                raise IndexError("Item with index '{}' does not exist".format(key))

    def __iter__(self):
        c = 0
        for k, v in self.data.items():
            while k > c:
                yield self._unset(c)
                c += 1

            yield v
            c += 1

    def __reversed__(self):
        l = len(self.data)
        return (self.data[i][1] for i in range(l - 1, -1, -1))

    def __add__(self, other):
        obj = self._clone()
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
        obj = self._clone()

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
        if not isinstance(n, int):
            raise TypeError("can't multiply sequence by non-int of type '{}'".format(type(n)))

        obj = self._clone()

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
        if not isinstance(n, int):
            raise TypeError("can't multiply sequence by non-int of type '{}'".format(type(n)))

        try:
            offset = self.tail() + 1
        except IndexError:
            offset = 0

        for c in range(offset, offset * n, offset):
            for i, v in self.data.items(stop=offset):
                self.data.insert(i + c, v)

        return self

    def __copy__(self):
        return self.copy()

    def insert(self, index, value):
        index = int(index)

        new = SkipList()
        for k, v in self.data.items(stop=index):
            new.insert(k, v)
        new.insert(index, value)
        for k, v in self.data.items(start=index):
            new.insert(k + 1, v)

        self.data = new

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
        l = len(self.data)
        for k1, k2 in zip(range(l // 2), range(l - 1, 0, -1)):
            self.data[k1], self.data[k2] = self.data[k2][1], self.data[k1][1]

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
        for k, v in enumerate(sorted(self.data.values())):
            self.data[k] = v

    def copy(self):
        obj = self._clone()
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

    def count(self, item):
        """
        Return total number of occurrences of given `item` in list
        :param item:
        """
        return len([1 for x in self.data.values() if x == item])

    def items(self, start=None, stop=None):
        if start is not None and start < 0:
            start = max(self.tail() + start + 1, 0)
        if stop is not None and stop < 0:
            stop = max(self.tail() + stop + 1, 0)

        return self.data.items(start=start, stop=stop)

    def keys(self, start=None, stop=None):  # NOQA
        """
        Return keys of non-empty items
        :param start:
        :param stop:
        :return:
        """
        return self.data.keys(start=start, stop=stop)

    def values(self, start=None, stop=None):  # NOQA
        """
        Return values of non-empty items
        :param start:
        :param stop:
        :return:
        """
        return self.data.values(start=start, stop=stop)

    def tail(self):
        """
        Return index of the last element
        :raises IndexError: no elements in list
        """
        return self.data[-1][0]

    def _slice_indexes(self, s):
        """
        Calculate positive index bounds from slice. If slice param is None, then it will be left as None
        :param s: slice object
        :return: start, stop, step
        """
        pieces = [s.start, s.stop, s.step]

        for i in [0, 1]:
            if pieces[i] is not None:
                pieces[i] = int(pieces[i])
                if pieces[i] < 0:
                    try:
                        last_ind = self.tail()  # IndexError if empty
                    except IndexError:
                        last_ind = 0
                    pieces[i] = max(last_ind + pieces[i] + 1, 0)

        if pieces[2] is not None:
            if pieces[2] < 0:
                raise ValueError('Negative slice step is not supported')
            elif pieces[2] == 0:
                raise ValueError('Slice step cannot be zero')

        return tuple(pieces)


__all__ = ('SparsedList', )
