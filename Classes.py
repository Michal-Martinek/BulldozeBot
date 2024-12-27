import heapq
import threading, queue

from enum import IntEnum, Enum, auto
from typing import Generic, TypeVar
import numpy as np
import numpy.typing as npt

TILES_TYPE = np.uint8
class Tiles(IntEnum):
	WALL = 0
	FREE = auto()
	ROCK = auto()
	BULLDOZER = auto()

Board = npt.NDArray

class Moves(Enum):
	UP = auto()
	RIGHT = auto()
	DOWN = auto()
	LEFT = auto()

	def offset(self):
		return (self == Moves.DOWN) - (self == Moves.UP), (self == Moves.RIGHT) - (self == Moves.LEFT)
	def orthogonalMoves(self) -> tuple:
		horizontal = (Moves.LEFT, Moves.RIGHT)
		return (Moves.UP, Moves.DOWN) if self in horizontal else horizontal

class Pos(tuple):
	@classmethod
	def _constructor(cls, y, x):
		return super().__new__(cls, (int(y), int(x)))
	def __new__(cls, y, x=None):
		if x is not None:
			y = (y, x)
		assert len(y) == 2, 'Two dimensions expected for Pos'
		return cls._constructor(*y)
	
	def __add__(self, other):
		if isinstance(other, Moves):
			other = other.offset()
		assert len(other) == 2
		return self._constructor(self[0] + other[0], self[1] + other[1])
	def __sub__(self, other):
		assert len(other) == 2
		return self._constructor(self[0] - other[0], self[1] - other[1])
	def __mul__(self, other):
		assert isinstance(other, int)
		return self._constructor(self[0] * other, self[1] * other)
	
	@classmethod
	def iterBoard(cls, b: Board, excludeOutmost=True):
		for y in range(excludeOutmost, b.shape[0] - excludeOutmost):
			for x in range(excludeOutmost, b.shape[1] - excludeOutmost):
				yield cls(y, x)
	@classmethod
	def iter3by3(cls, center, *, includeCenter=False, crossOnly=False):
		for y in range(center[0] - 1, center[0] + 2):
			for x in range(center[1] - 1, center[1] + 2):
				pos = cls(y, x)
				if crossOnly and y != center[0] and x != center[1]: continue
				if not includeCenter and pos == center: continue
				yield cls(y, x)
	@classmethod
	def getCoordsWhere(cls, cond, onlyOne=False) -> list:
		a = np.array(np.where(cond)).T
		a = [cls(p) for p in a]
		if onlyOne:
			assert len(a) == 1
			a = a[0]
		return a

class Comms:
	"""class for inter-Thread communication between Main and Solve threads"""
	def __init__(self):
		self.stateQueue = queue.Queue()
		self.doneEvent = threading.Event()
		self.stopEvent = threading.Event()
		self.movesQueue: queue.Queue[list[Moves]] = queue.Queue()

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
