"""
Programmed by Austin Dunn
"""

class PriorityQueue:

    def __init__(self):
        self.heap = [None]*20
        self.size = 0

    def get_heap(self):
        return [i for i in self.heap if i is not None]

    def add(self, node):
        if node is None:
            raise ValueError('tried to add null object')

        if self.size > len(self.heap) // 2:
            self.grow()

        if self.size == 0:
            self.heap[1] = node
            self.cache_size += node.get_size()
            self.size += 1
        else:
            self.size += 1
            self.cache_size += node.get_size()
            self.heap[self.size] = node
            self.up_heap()

    def pop(self):
        if self.size == 0:
            raise ValueError('queue is empty')

        removed_key = self.heap[1].get_key()
        self.cache_size -= self.heap[1].get_size()

        self.heap[1] = self.heap[self.size]
        self.heap[self.size] = None
        self.size -= 1
        self.down_heap(1)
        return removed_key

    def remove(self, i):
        if self.size == 0:
            raise RuntimeError('queue is empty')

        self.cache_size -= self.heap[i].get_size()

        self.heap[i] = self.heap[self.size]
        self.heap[self.size] = None
        self.size -= 1
        self.down_heap(i)

    def swap_root(self, node):
        if node is None:
            raise ValueError('tried to swap invalid object')

        self.heap[1] = node

        return self.down_heap(1)

    def peek(self):
        return self.heap[1]

    def contains(self, key):
        for node in self.heap:
            if node is not None:
                if node.get_key() == key:
                    return True
        return False

    def update_key(self, i):

        # update node values
        self.heap[i].update_value()

        # swap key downwards if needed
        return self.down_heap(i)

    def grow(self):
        temp_list = [None] * len(self.heap) * 2
        for i in range(self.size + 1):
            temp_list[i] = self.heap[i]
        self.heap = temp_list

    def up_heap(self):
        i = self.size

        while (i > 1 and self.heap[i // 2].get_value() > self.heap[i].get_value()):
            self.heap[i] , self.heap[i // 2] = (self.heap[i // 2], self.heap[i])
            i = i // 2

    def down_heap(self, i, new_mapping):
        while (2 * i < (2 * self.size) + 1 and (self.heap[2 * i] is not None or self.heap[(2 * i) + 1] is not None)):
            j = (2 * i) + 1
            if self.heap[2 * i] is not None and self.heap[(2 * i) + 1] is not None:
                if (self.heap[2 * i].get_value() < self.heap[(2 * i) + 1].get_value()):
                    j = 2 * i
            elif self.heap[2 * i] is not None and self.heap[(2 * i) + 1] is None:
                j = 2 * i
            if self.heap[i].get_value() > self.heap[j].get_value():
                self.heap[i] , self.heap[j] = (self.heap[j], self.heap[i])
            i = j

class PriorityNode:

    def __init__(self, time, ref, denom, weight, key):
        self.t = time
        self.r = ref
        self.value = (time * ref) / denom
        self.weight = weight
        self.key = key

    def update_value(self):
        self.r += 1
        self.value = self.r * self.t

    def get_value(self):
        return self.value

    def get_key(self):
        return self.key

    def get_time(self):
        return self.t

    def get_ref(self):
        return self.r

    def __lt__(self, other): # For x < y
        if not isinstance(other, PriorityNode):
            return False
        return self.value < other.value
    def __le__(self, other): # For x <= y
        if not isinstance(other, PriorityNode):
            return False
        return self.value <= other.value
    def __eq__(self, other): # For x == y
        if not isinstance(other, PriorityNode):
            return False
        return self.value == other.value
    def __ne__(self, other): # For x != y
        if not isinstance(other, PriorityNode):
            return False
        return self.value != other.value
    def __gt__(self, other): # For x > y
        if not isinstance(other, PriorityNode):
            return False
        return self.value > other.value
    def __ge__(self, other): # For x >= y
        if not isinstance(other, PriorityNode):
            return False
        return self.value >= other.value
