sparsedlist
===========

**sparsedlist** is endless list with non-contiguous indexes. Based on
`Skip list <https://en.wikipedia.org/wiki/Skip_list>`__ data structure.
Python3 is used.

**sparsedlist** is a list structure, where set of indexes can have
“gaps” and you can put a value to any index. In other words, the
structure is similar to dict, but has list interface and sorted numeric
indexes.

Since *skiplist* structure is used as machinery, then you have fast
forward iteration with O(1) complexity and pretty good
indexation/insertion/deletion with O(log n) complexity.

Example:

.. code:: python

   >>> from sparsedlist import SparsedList
   >>> s = SparsedList()
   >>> s[180] = 'rock the microphone'
   >>> s[10:20] = range(10)
   >>> print(s)
   SparsedList{{10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 180: 'rock the microphone'}}
   >>> print(s[180])
   rock the microphone
   >>> print(s[-1])
   rock the microphone
   >>> print(s[-2])
   None
   >>> print(list(s[18:23]))
   [8, 9, None, None, None]
   >>> print(s[100500])
   None

By default **sparsedlist** substitutes item on **None** value if it has
not set. To disable this feature, pass *required* param to constructor.
Then **IndexError** will be raised on getting unset items. For instance:

.. code:: python

   >>> from sparsedlist import SparsedList
   >>> s = SparsedList(required=True)
   >>> s[10:20] = range(10)
   >>> print(s[100500])
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "/usr/local/lib/python3.6/dist-packages/sparsedlist.py", line 73, in __getitem__
       raise IndexError("Item with index '{}' does not exist".format(item))
   IndexError: Item with index '100500' does not exist
   >>> print(list(s[18:25]))
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
     File "/usr/local/lib/python3.6/dist-packages/sparsedlist.py", line 73, in __getitem__
       raise IndexError("Item with index '{}' does not exist".format(item))
   IndexError: Item with index '20' does not exist

Dependencies
============

``pyskiplist`` only. Tested on ``python3.6``.

Installation
============

::

   pip3 install sparsedlist

Author
======

Igor Derkach, gosha753951@gmail.com
