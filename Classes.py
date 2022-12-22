import heapq
from enum import IntEnum, auto
from typing import Generic, TypeVar
from dataclasses import dataclass

class Tiles(IntEnum):
	WALL = auto()
	FREE = auto()
	ROCK = auto()
	BULLDOZER = auto()

class Moves: # 10 * (x off + 1) + (y off + 1)
	UP = 10
	DOWN = 12
	RIGHT = 21
	LEFT = 1

BoardDtype = TypeVar('BoardDtype')
@ dataclass
class Pos:
	x: int
	y: int

	def moved(self, move: Moves):
		return Pos(self.x + (move // 10) - 1, self.y + (move %  10) - 1)
	def move(self, move: Moves):
		self.x += (move // 10) - 1
		self.y += (move %  10) - 1
	def copy(self):
		return Pos(self.x, self.y)
	def at(self, b: list[list[BoardDtype]]) -> BoardDtype:
		return b[self.y][self.x]
	def __hash__(self):
		return hash((self.x, self.y))
	def __iter__(self):
		return iter((self.x, self.y))
	@ classmethod
	def iterBoard(cls, b: list[list], inner=True):
		for y in range(inner, len(b)-inner):
			for x in range(inner, len(b[0])-inner):
				yield cls(x, y)
	def __eq__(self, other):
		return self.x == other.x and self.y == other.y
	def __lt__(self, other):
		return self.x < other.x and self.y < other.y
	def __repr__(self):
		return f'P({self.x},{self.y})'

T = TypeVar('T')
class Heap(Generic[T]):
	def __init__(self, item:T=None):
		self.heap: list[T] = []
		self.inSet: set[T] = set()
		if item is not None:
			self.push(item)
	def __repr__(self) -> str:
		return 'Heap(' + repr(self.heap) + ')'
	
	def push(self, item):
		assert getattr(item, '__lt__', None), 'the item must have a __lt__ method'
		assert getattr(item, '__hash__', None), 'the item must have a __hash__ method'
		assert getattr(item, '__eq__', None), 'the item must have a __eq__ method'
		assert item not in self.inSet
		heapq.heappush(self.heap, item)
		self.inSet.add(item)
	def index(self, key):
		'''use only if the item is a (priority, key) tuple'''
		return list(map(lambda x:x[1], self.heap)).index(key)
	def __getitem__(self, key):
		return self.heap[self.index(key)]
	def getItemIdx(self, item):
		'''returns the index of item inside itself which is __eq__ to param item'''
		return self.heap.index(item)
	def __len__(self):
		return len(self.heap)
	def __bool__(self):
		return bool(len(self.heap))

	def pop(self):
		item = heapq.heappop(self.heap)
		self.inSet.remove(item)
		return item
	def decreasePriority(self, key, priority):
		'''use only if the item is a (priority, key) tuple'''
		i = self.index(key)
		self.inSet.remove(self.heap[i])
		self.heap[i] = (priority, *self.heap[i][1:])
		self.inSet.add(self.heap[i])
		self._siftdown(i)
	def changeMetadata(self, i, new: tuple):
		'''use only if the item is a (priority, key, meta1, ...) tuple'''
		self.inSet.remove(self.heap[i])
		self.heap[i] = (*self.heap[i][:2], *new)
		self.inSet.add(self.heap[i])
	def changedPriority(self, i):
		self._siftdown(i)
	
	def hasItem(self, item) -> bool:
		return item in self.inSet
	def hasKey(self, key) -> bool:
		try:
			self.index(key)
			return True
		except ValueError:
			return False
	def _siftdown(self, pos):
		newitem = self.heap[pos]
		# Follow the path to the root, moving parents down until finding a place
		# newitem fits.
		while pos > 0:
			parentpos = (pos - 1) >> 1
			parent = self.heap[parentpos]
			if newitem < parent:
				self.heap[pos] = parent
				pos = parentpos
				continue
			break
		self.heap[pos] = newitem
