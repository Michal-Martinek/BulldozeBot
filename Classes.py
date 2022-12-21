import heapq # TODO: make better heap, with decrease key and test for in
from enum import IntEnum, auto
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
	def at(self, b: list[list]):
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
	# def __iadd__(self, other):
	# 	if isinstance(other, Pos):
	# 		self.x += other.x
	# 		self.y += other.y
	# 	elif isinstance(other, tuple):
	# 		assert len(other) == 2
	# 		self.x += other[0]
	# 		self.y += other[1]
	# 	else: assert False

class Heap:
	pass