# sparsedlist

**sparsedlist** is endless list with non-contiguous indexes. Based on [Skip list](https://en.wikipedia.org/wiki/Skip_list) structure.

To add value to the list just set it with any index:

```
>>> from sparsedlist import SparsedList
>>> s = SparsedList()
>>> s[180] = 'just set it'
>>> s[10:20] = range(10)
>>> print(s)
SparsedList{{10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 180: 'just set it'}}
>>> print(s[180])
just set it
```

Since *skiplist* structure is used, then you have fast iteration with O(1) complexity and pretty good indexation with O(log(n)) complexity, where n is count of items in list.