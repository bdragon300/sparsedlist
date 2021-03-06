import pytest
import random
import itertools
from sparsedlist import SparsedList
from pyskiplist import SkipList
import re
import copy


machinery_class = SkipList


@pytest.fixture
def plain_data():
    return [(i, object()) for i in range(40)]


@pytest.fixture
def powertwo_data():
    return [(2 ** i, object()) for i in range(10)]


def slice_permutations(start, stop, step):
    z = (list(zip(x, (start, stop, step))) for x in itertools.product((None, 1), repeat=3))
    for i in z:
        p = list(None if j[0] is None else j[1] for j in i)
        yield slice(*p)


class TestSparsedList:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.obj = SparsedList()

    def test_props_default_values(self):
        assert isinstance(self.obj.data, machinery_class) and len(self.obj.data) == 0

    def test_initlist(self):
        test_data = [random.randrange(1000) for x in range(100)]
        check_data = enumerate(test_data)

        obj = SparsedList(initlist=test_data)

        assert all(a == b for a, b in itertools.zip_longest(check_data, obj.items()))

    def test_inititems(self):
        test_data = [(1, 4), (11, 10), (200, 0)]

        obj = SparsedList(inititems=test_data)

        assert list(obj.items()) == test_data

    def test_repr(self, plain_data):
        self.obj.extend(plain_data)
        assert re.match('SparsedList', repr(self.obj))

    @pytest.mark.parametrize('a,b,expect', (
        ({0: 0, 5: 5, 10: 10}, {0: 0, 5: 5, 10: 10}, True),
        ({0: 0, 5: 5, 10: 10}, {0: 0, 5: 5}, False),
        ({0: 0, 6: 6, 10: 10}, {0: 0, 5: 5, 10: 10}, False),
        ({0: 0, 5: 6, 10: 10}, {0: 0, 5: 5, 10: 10}, False)
    ))
    def test_equal(self, a, b, expect):
        obj = SparsedList()
        obj.extend(a.items())
        self.obj.extend(b.items())

        res = self.obj == obj

        assert res is expect

    @pytest.mark.parametrize('a,b,expect', (
        ({0: 0, 5: 5, 10: 10}, {0: 0, 5: 5, 10: 10}, False),
        ({0: 0, 5: 5, 10: 10}, {0: 0, 5: 5}, True),
        ({0: 0, 6: 6, 10: 10}, {0: 0, 5: 5, 10: 10}, True),
        ({0: 0, 5: 6, 10: 10}, {0: 0, 5: 5, 10: 10}, True)
    ))
    def test_not_equal(self, a, b, expect):
        obj = SparsedList()
        obj.extend(a.items())
        self.obj.extend(b.items())

        res = self.obj != obj

        assert res is expect

    def test_contains(self, powertwo_data):
        self.obj.extend(powertwo_data)
        d = dict(powertwo_data)

        res = d[2 ** 3] in self.obj

        assert res

    def test_contains2(self, powertwo_data):
        self.obj.extend(powertwo_data)
        d = dict(powertwo_data)

        res = object in self.obj

        assert not res

    def test_len(self, powertwo_data):
        self.obj.extend(powertwo_data)
        check_data = len(powertwo_data)

        res = len(self.obj)

        assert res == check_data

    @pytest.mark.parametrize('ind', (1, -2))
    def test_getitem_index(self, ind, plain_data):
        self.obj.extend(plain_data)
        test_data = [i[1] for i in plain_data]

        res = self.obj[ind]

        assert res == test_data[ind]

    @pytest.mark.parametrize('ind', (3, 2**15))
    def test_getitem_return_none_on_unset_element(self, ind, powertwo_data):
        self.obj.extend(powertwo_data)

        res = self.obj[ind]

        assert res is None

    @pytest.mark.parametrize('ind', (
        slice(1, 10), slice(None, 10), slice(1, 10, 3), slice(None, 10, 3),
        slice(-10, -1), slice(None, -1), slice(-10, -1, 3), slice(None, -1, 3)
    ))
    def test_getitem_slice(self, ind, plain_data):
        self.obj.extend(plain_data)
        check_data = [i[1] for i in plain_data]

        res = self.obj[ind]

        assert list(res) == check_data[ind]

    @pytest.mark.parametrize('ind', (
        slice(1, None), slice(1, None, 3),
        slice(-10, None), slice(-10, None, 3)
    ))
    def test_getitem_slice_endless(self, ind, plain_data):
        self.obj.extend(plain_data)
        limit = 100
        check_data = [i[1] for i in plain_data][ind]
        check_data += [None] * (limit - len(check_data))

        gen = self.obj[ind]
        res = [next(gen) for x in range(0, limit)]

        assert res == check_data

    @pytest.mark.parametrize('ind', (
        slice(1, 10), slice(1, 10, 3),
        slice(None, 10), slice(None, 10, 3)
    ))
    def test_getitem_slice_nones_on_empty_list(self, ind):
        check_data = [None for i in range(*ind.indices(2 ** 10))]

        res = list(self.obj[ind])

        assert res == check_data

    @pytest.mark.parametrize('ind', (
        slice(5, 5), slice(5, 5, 2),
        slice(100, 100), slice(100, 100, 3),
        slice(10, 5), slice(10, 5, 3),
        slice(-200, -100), slice(-200, -100, 3)
    ))
    def test_getitem_slice_return_empty_iterable(self, ind, plain_data):
        self.obj.extend(plain_data)

        res = self.obj[ind]

        assert list(res) == []

    @pytest.mark.parametrize('ind', (
        slice(2 ** 15, 2 ** 16), slice(None, 2 ** 16), slice(2 ** 15, 2 ** 16, 3), slice(None, 2 ** 16, 3),
        slice(2, 50), slice(None, 50), slice(2, 50, 3), slice(None, 50, 3),
        slice(-8, -1), slice(None, -1), slice(-8, -1, 3), slice(None, -1, 3)
    ))
    def test_getitem_slice_fill_gaps_with_nones(self, ind, powertwo_data):
        self.obj.extend(powertwo_data)
        d = dict(powertwo_data)
        dlen = sorted(d.keys())[-1] + 1
        start = dlen + ind.start if (ind.start and ind.start < 0) else ind.start or 0
        stop = dlen + ind.stop if (ind.stop and ind.stop < 0) else ind.stop or dlen
        check_data = [
            d[i] if i in d else None
            for i in range(start, stop, ind.step or 1)
        ]
        # check_data = [powertwo_data[i] if i in powertwo_data else None for i in range(*ind.indices(2 ** 9 + 1))]

        res = list(self.obj[ind])

        assert res == check_data

    def test_getitem_error_too_large_negative_index(self, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(IndexError):
            res = self.obj[-100500]

    @pytest.mark.parametrize('step', (0, -1))
    def test_getitem_error_on_zero_or_negative_step(self, step, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(ValueError):
            res = self.obj[1:3:step]

    @pytest.mark.parametrize('ind', (
        slice(1, 10), slice(None, 10), slice(1, 10, 3), slice(None, 10, 3)
    ))
    def test_getitem_required_error_on_empty_list(self, ind):
        obj = SparsedList(required=True)

        with pytest.raises(IndexError):
            list(obj[ind])

    @pytest.mark.parametrize('ind', (3, 2**15))
    def test_getitem_required_error_on_unset_element_get(self, ind, powertwo_data):
        obj = SparsedList(required=True)
        obj.extend(powertwo_data)

        with pytest.raises(IndexError):
            res = obj[ind]

    def test_getitem_required_error_too_large_negative_index(self, plain_data):
        obj = SparsedList(required=True)
        obj.extend(plain_data)

        with pytest.raises(IndexError):
            res = obj[-100500]

    @pytest.mark.parametrize('ind', (
        slice(2 ** 15, 2 ** 16), slice(None, 2 ** 16), slice(2 ** 15, 2 ** 16, 3), slice(None, 2 ** 16, 3),
        slice(2, 50), slice(None, 50), slice(2, 50, 3), slice(None, 50, 3),
        slice(-200, -1), slice(None, -1), slice(-200, -1, 3), slice(None, -1, 3)
    ))
    def test_getitem_required_error_on_unset_elements_in_slice_stop_is_not_none(self, ind, powertwo_data):
        if ind.stop is None:
            pytest.skip()

        obj = SparsedList(required=True)
        obj.extend(powertwo_data)

        with pytest.raises(IndexError):
            list(obj[ind])

    def test_getitem_required_empty_iterable_when_slice_too_far_left(self, plain_data):
        obj = SparsedList(required=True)
        obj.extend(plain_data)

        res = obj[-200:-100]

        assert list(res) == []

    @pytest.mark.parametrize('ind', (slice(5, None), slice(None)))
    def test_getitem_required_error_eventually_occurs_on_endless_slice(self, ind, plain_data):
        obj = SparsedList(required=True)
        obj.extend(plain_data)

        with pytest.raises(IndexError):
            assert list(obj[ind])

    @pytest.mark.parametrize('ind', (2, -3, 1000))
    def test_setitem_index(self, ind, plain_data):
        self.obj.extend(plain_data)

        self.obj[ind] = 'test_mark'

        assert self.obj[ind] == 'test_mark'

    @pytest.mark.parametrize('ind', (2, -3, 1000))
    def test_setitem_index_replace_old_value(self, ind, plain_data):
        self.obj.extend(plain_data)

        self.obj[ind] = 'test'
        self.obj[ind] = 'test_mark'

        assert self.obj[ind] == 'test_mark'
        assert len(list(self.obj.items(start=ind, stop=ind + 1))) == 1  # SkipList allows add several value for one key

    @pytest.mark.parametrize('ind', itertools.chain(
        slice_permutations(0, 10, 3),
        slice_permutations(100, 110, 3)
    ))
    def test_setitem_slice(self, ind):
        list_length = 10
        test_data = ['test_mark' + str(x) for x in range(list_length)]

        self.obj[ind] = test_data

        assert all(a == b for a, b in zip(self.obj[ind], test_data))

    @pytest.mark.parametrize('ind', (
        slice(0, 5),
        slice(None, 5),
        slice(0, 5, 2),
        slice(-11, -5),
        slice(-11, -5, 2)
    ))
    def test_setitem_cut_value_if_slice_shorter_on_filled_list(self, ind, plain_data):
        self.obj.extend(plain_data)
        list_length = 10
        test_data = ['test_mark'] * list_length

        self.obj[ind] = test_data

        assert self.obj[ind.stop + ((ind.step or 1) - 1)] != 'test_mark'  # Check next item

    @pytest.mark.parametrize('ind', (
        slice(0, 5),
        slice(None, 5),
        slice(0, 5, 2)
    ))
    def test_setitem_drop_the_rest_iterable_elements_if_slice_is_shorter(self, ind):
        list_length = 10
        test_data = ['test_mark'] * list_length

        self.obj[ind] = test_data

        # Check if next item is unset in list
        check_index = ind.stop + ((ind.step or 1) - 1)
        assert check_index not in self.obj.keys()

    @pytest.mark.parametrize('ind', slice_permutations(5, 25, 3))
    def test_setitem_remove_the_rest_elements_if_positive_slice_wider_than_iterable(self, ind, plain_data):
        self.obj.extend(plain_data)
        check_data = list(plain_data)
        list_length = len(check_data) - 10
        test_data = ['test_mark'] * list_length

        self.obj[ind] = test_data

        unset_start = (ind.start or 0) + list_length * (ind.step or 1)
        assert all(i not in self.obj.keys() for i in range(unset_start, ind.stop or len(check_data), ind.step or 1))

    @pytest.mark.parametrize('ind', slice_permutations(-25, -5, 3))
    def test_setitem_delete_the_rest_elements_if_negative_slice_wider_than_iterable(self, ind, plain_data):
        self.obj.extend(plain_data)
        check_data = list(plain_data)
        list_length = len(check_data) - 10
        test_data = ['test_mark'] * list_length

        self.obj[ind] = test_data

        # Convert negative indexes to positive ones
        unset_start = (len(check_data) + (ind.start or 0)) + list_length * (ind.step or 1)
        unset_stop = len(check_data) + (ind.stop or 0)
        assert all(i not in self.obj.keys() for i in range(unset_start, unset_stop, ind.step or 1))

    def test_setitem_error_if_slice_argument_is_not_iterable(self):
        with pytest.raises(TypeError):
            self.obj[0:2] = 123

    def test_setitem_indexerror_too_much_negative_int_index(self, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(IndexError):
            self.obj[-1000] = 123

    def test_setitem_error_negative_index_on_empty_list(self):
        with pytest.raises(IndexError):
            self.obj[-1] = 'test_mark'

    @pytest.mark.parametrize('ind', (
        slice(5, 5), slice(5, 5, 3),
        slice(-200, -100), slice(-200, -100, 3)
    ))
    def test_setitem_do_nothing_on_slice_set_on_empty_list(self, ind):
        self.obj[ind] = range(100)

        assert len(self.obj) == 0

    @pytest.mark.parametrize('step', (0, -1))
    def test_setitem_error_on_zero_or_negative_step(self, step, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(ValueError):
            self.obj[1:3:step] = [1, 2]

    @pytest.mark.parametrize('ind', (2, -2))
    def test_delitem(self, ind, plain_data):
        self.obj.extend(plain_data)

        del self.obj[ind]

        assert self.obj[ind] is None

    @pytest.mark.parametrize('ind', itertools.chain(
        slice_permutations(0, 5, 3),
        slice_permutations(0, 200, 3),
        slice_permutations(-10, -5, 3),
        slice_permutations(5, 5, 3)
    ))
    def test_delitem_slice(self, ind, plain_data):
        self.obj.extend(plain_data)

        del self.obj[ind]

        assert all(i not in self.obj for i in range(*ind.indices(len(plain_data))))

    @pytest.mark.parametrize('ind', (1000, -1000))
    def test_delitem_error(self, ind, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(IndexError):
            del self.obj[ind]

    def test_iter_error_on_unset(self, powertwo_data):
        self.obj.extend(powertwo_data)
        d = dict(powertwo_data)
        check_data = [d[i] if i in d else None for i in range(sorted(d.keys())[-1] + 1)]

        assert list(self.obj) == check_data

    def test_iter_required_error_on_unset(self, powertwo_data):
        obj = SparsedList(required=True)
        obj.extend(powertwo_data)
        check_data = 3

        with pytest.raises(IndexError) as e:
            c = 0
            for i in obj:
                c += 1

            assert c == check_data

    def test_reversed_result(self, powertwo_data):
        self.obj.extend(powertwo_data)
        check_data = [i[1] for i in reversed(powertwo_data)]

        res = reversed(self.obj)

        assert list(res) == check_data

    def test_reversed_not_modify(self, powertwo_data):
        self.obj.extend(powertwo_data)

        res = reversed(self.obj)

        assert list(self.obj.items()) == powertwo_data

    def test_add_list(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = [1, 5, 10, 100]
        check_data = list(powertwo_data)
        check_data += enumerate(test_data, start=check_data[-1][0] + 1)

        res = self.obj.__add__(test_data)

        assert list(res.items()) == check_data
        assert res is not self.obj

    def test_add_object(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = SparsedList(inititems=powertwo_data)
        check_data = list(powertwo_data)
        offset = check_data[-1][0] + 1
        check_data += [(x[0] + offset, x[1]) for x in powertwo_data]

        res = self.obj.__add__(test_data)

        assert list(res.items()) == check_data
        assert res is not self.obj

    def test_radd_list(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = [1, 5, 10, 100]
        check_data = list(enumerate(test_data)) + [(x[0] + len(test_data), x[1]) for x in powertwo_data]

        res = self.obj.__radd__(test_data)

        assert list(res.items()) == check_data
        assert res is not self.obj

    def test_radd_object(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = SparsedList(inititems=powertwo_data)
        check_data = list(powertwo_data)
        offset = check_data[-1][0] + 1
        check_data += [(x[0] + offset, x[1]) for x in powertwo_data]

        res = self.obj.__radd__(test_data)

        assert list(res.items()) == check_data
        assert res is not self.obj

    def test_iadd_list(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = [1, 5, 10, 100]
        check_data = list(powertwo_data)
        check_data += enumerate(test_data, start=check_data[-1][0] + 1)

        res = self.obj.__iadd__(test_data)

        assert list(res.items()) == check_data
        assert res is self.obj

    def test_iadd_object(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = SparsedList(inititems=powertwo_data)
        check_data = list(powertwo_data)
        offset = check_data[-1][0] + 1
        check_data += [(x[0] + offset, x[1]) for x in powertwo_data]

        res = self.obj.__iadd__(test_data)

        assert list(res.items()) == check_data
        assert res is self.obj

    def test_mul(self, powertwo_data):
        self.obj.extend(powertwo_data)
        n = 4
        offset = list(powertwo_data)[-1][0] + 1
        check_data = [(x[0] + offset * c, x[1]) for c in range(n) for x in powertwo_data]

        res = self.obj.__mul__(n)

        assert list(res.items()) == check_data
        assert res is not self.obj

    @pytest.mark.parametrize('n', (None, (1,2,3), 1.1))
    def test_mul_accepts_int_only(self, powertwo_data, n):
        self.obj.extend(powertwo_data)

        with pytest.raises(TypeError):
            self.obj.__mul__(n)

    def test_imul(self, powertwo_data):
        self.obj.extend(powertwo_data)
        n = 4
        offset = list(powertwo_data)[-1][0] + 1
        check_data = [(x[0] + offset * c, x[1]) for c in range(n) for x in powertwo_data]

        res = self.obj.__imul__(n)

        assert list(res.items()) == check_data
        assert res is self.obj

    @pytest.mark.parametrize('n', (None, (1,2,3), 1.1))
    def test_imul_accepts_int_only(self, powertwo_data, n):
        self.obj.extend(powertwo_data)

        with pytest.raises(TypeError):
            self.obj.__imul__(n)

    def test_copycopy(self, plain_data):
        self.obj.extend(plain_data)

        res = copy.copy(self.obj)

        assert res is not self.obj \
               and res.data is not self.obj.data \
               and all(res[x] is self.obj[x] for x in self.obj.keys())

    def test_copydeepcopy(self, plain_data):
        self.obj.extend(plain_data)

        res = copy.deepcopy(self.obj)

        assert res is not self.obj \
               and res.data is not self.obj.data \
               and all(res[x] is not self.obj[x] for x in self.obj.keys())

    def test_insert1(self, plain_data):
        self.obj.extend(plain_data)
        test_data = object()
        check_data = plain_data[:10] \
                     + [(10, test_data)] \
                     + [(k + 1, v) for k, v in plain_data[10:]]

        self.obj.insert(10, test_data)

        assert list(self.obj.items()) == check_data

    def test_insert2(self, powertwo_data):
        self.obj.extend(powertwo_data)
        test_data = object()
        check_data = [(k, v) for k, v in powertwo_data if k < 10] \
                     + [(10, test_data)] \
                     + [(k + 1, v) for k, v in powertwo_data if k >= 10]

        self.obj.insert(10, test_data)

        assert list(self.obj.items()) == check_data

    def test_append(self, plain_data):
        self.obj.extend(plain_data)
        test_data = 'test_mark'

        self.obj.append(test_data)

        assert self.obj[-1] == test_data

    def test_extend(self, powertwo_data, plain_data):
        check_data = dict(powertwo_data)
        check_data.update(plain_data)
        check_data = sorted(check_data.items())

        self.obj.extend(powertwo_data)
        self.obj.extend(plain_data)

        assert all(a[0] == b[0] and a[1] is b[1] for a, b in zip(self.obj.items(), check_data))

    def test_clear(self, plain_data):
        self.obj.extend(plain_data)

        self.obj.clear()

        assert len(self.obj) == 0

    def test_reverse1(self, plain_data):
        self.obj.extend(plain_data)

        self.obj.reverse()

        assert list(self.obj.items()) == list(zip((i[0] for i in plain_data), (i[1] for i in plain_data[::-1])))

    def test_reverse2(self, powertwo_data):
        self.obj.extend(powertwo_data)

        self.obj.reverse()
        assert list(self.obj.items()) == list(zip((i[0] for i in powertwo_data), (i[1] for i in powertwo_data[::-1])))

    @pytest.mark.parametrize('ind', (-2, 1))
    def test_pop(self, ind, plain_data):
        self.obj.extend(plain_data)
        *_, check_data = zip(*list(plain_data))

        res = self.obj.pop(ind)

        assert res == check_data[ind]
        assert len(self.obj) == len(check_data) - 1

    def test_pop_error_empty_list(self):
        with pytest.raises(IndexError):
            self.obj.pop()

    def test_remove(self, plain_data):
        self.obj.extend(plain_data)
        *_, test_data = zip(*list(plain_data))

        self.obj.remove(test_data[10])

        with pytest.raises(ValueError):
            self.obj.index(test_data[10])
        assert list(self.obj[:10]) == list(test_data[:10]) and list(self.obj[11:self.obj.tail() + 1]) == list(test_data[11:])

    def test_remove_error_not_found(self, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(ValueError):
            self.obj.remove(123)

    def test_sort1(self):
        integers = [i for i in range(10)]
        check_data = list(enumerate(integers))
        random.shuffle(integers)
        shuffled = list(enumerate(integers))
        self.obj.extend(shuffled)

        self.obj.sort()

        assert list(self.obj.items()) == check_data

    def test_sort2(self):
        integers = [i for i in range(10)]
        check_data = list(zip((2 ** i for i in range(10)), integers))
        random.shuffle(integers)
        shuffled = list(zip((2 ** i for i in range(10)), integers))
        self.obj.extend(shuffled)

        self.obj.sort()

        assert list(self.obj.items()) == check_data

    def test_copy(self, plain_data):
        self.obj.extend(plain_data)

        res = self.obj.copy()

        assert res is not self.obj
        assert list(self.obj.items()) == list(res.items())

    @pytest.mark.parametrize('start,stop', (
        (None, None),
        (None, 20),
        (10, None),
        (10, 20)
    ))
    def test_index_start_stop(self, start, stop, plain_data):
        self.obj.extend(plain_data)
        *_, test_data = zip(*list(plain_data))
        ind = 15

        res = self.obj.index(test_data[ind], start=start, stop=stop)

        assert res == ind

    @pytest.mark.parametrize('start,stop,ind', (
        (None, 20, 20),
        (10, None, 9),
        (10, 20, 20)
    ))
    def test_index_error_not_found_start_stop(self, start, stop, ind, plain_data):
        self.obj.extend(plain_data)
        test_data, *_ = zip(*list(plain_data))

        with pytest.raises(ValueError):
            self.obj.index(test_data[ind], start=start, stop=stop)

    def test_index_error_not_found(self, plain_data):
        self.obj.extend(plain_data)

        with pytest.raises(ValueError):
            self.obj.index('not_found')

    def test_count_item_arg(self):
        test_data = enumerate([x % 5 for x in range(40)])
        check_data = len([x for x in test_data if x == 1])
        self.obj.extend(test_data)

        res = self.obj.count(1)

        assert res == check_data

    @pytest.mark.parametrize('start,stop', (
        (None, None),
        (None, 20),
        (10, None),
        (10, 20)
    ))
    def test_items(self, start, stop, powertwo_data):
        self.obj.extend(powertwo_data)
        check_data = \
            list(x for x in powertwo_data if (start is None or x[0] >= start) and (stop is None or x[0] < stop))

        assert list(self.obj.items(start=start, stop=stop)) == check_data

    def test_tail(self, plain_data):
        self.obj.extend(plain_data)
        check_data = list(plain_data)[-1][0]

        res = self.obj.tail()

        assert res == check_data

    def test_tail_error_if_empty(self):
        with pytest.raises(IndexError):
            self.obj.tail()
