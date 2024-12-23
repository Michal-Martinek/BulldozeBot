from collections import deque
import numpy as np
from Classes import *

class State:
	tiles: np.ndarray
	targets: set[Pos]
	forbidden: np.ndarray=None
	distMaps: np.ndarray=None
	
	@classmethod
	def globalInit(cls, rocks: set[Pos], bulldozerPos: Pos, tiles: Board, targets: set[Pos], forbidden: Board):
		cls.tiles = tiles
		cls.targets = targets
		cls.forbidden = forbidden
		return cls(rocks, bulldozerPos)

	def __init__(self, rocks: set[Pos], bulldozerPos: Pos, moves: list[Moves]=[]):
		self.rocks: set[Pos] = rocks
		self.bulldozerPos = bulldozerPos
		self.moves: list[Moves] = moves

		self._hCost = -1
		self._toppest = None
	def levelWon(self) -> bool:
		return self.rocks == self.targets
	def movedBulldozer(self, move: Moves):
		return State(self.rocks.copy(), self.bulldozerPos + move, self.moves + [move])
	def adjustMovedRock(self, newRockPos: Pos):
		self.rocks.remove(self.bulldozerPos)
		self.rocks.add(newRockPos)
		self._toppest = None
		self._hCost = -1
	def takeBetterFrom(self, other):
		if len(self.moves) > len(other.moves): # TODO: the two states don't have to have the same bulldozer pos, so the moves couldn't be the same
			self.moves = other.moves.copy()
		raise NotImplementedError

	def __hash__(self) -> int:
		return hash((tuple(self.rocks), self.toppest))
	def __repr__(self) -> str:
		c = self.cost
		return f'State({self.bulldozerPos}, g={c}, h={self._hCost}, rocks={self.rocks})'
	def __lt__(self, other):
		if self.cost == other.cost:
			return self._hCost < other._hCost
		return self.cost < other.cost
	def __eq__(self, other):
		return self.rocks == other.rocks and self.toppest == other.toppest
	def _getToppest(self) -> Pos:
		toppest = Pos(self.tiles.shape)
		closed = set((self.bulldozerPos, ))
		opened = deque((self.bulldozerPos, ))
		while opened:
			curr = opened.popleft()
			if curr < toppest:
				toppest = curr
			for move in Moves:
				new = curr + move
				if self.tiles[new] == Tiles.FREE and new not in self.rocks and new not in closed:
					opened.append(new)
					closed.add(new)
		assert toppest < self.tiles.shape
		return toppest
	
	def calcHeuristic(self) -> bool:
		table = np.zeros((len(self.rocks), len(self.targets)), dtype='float32')
		for r, rock in enumerate(self.rocks):
			for m, distMap in enumerate(self.distMaps):
				cost = distMap[rock]
				table[r, m] = cost
		self._hCost = self._bestHeuristic(table)
		return self._hCost != np.float32('inf')
	def _bestHeuristic(self, table: np.ndarray):
		if table.size == 1:
			return table[0, 0]
		for col in np.argsort(table[0]):
			if table[0, col] == float('inf'): break # there are only infs left
			otherColIdxs = np.delete(np.arange(table.shape[1]), col)
			cost = self._bestHeuristic(table[1:, otherColIdxs])
			if cost == float('inf'): continue
			return cost + table[0][col]
		return float('inf')

	@ property
	def toppest(self) -> Pos:
		if self._toppest is not None:
			return self._toppest
		self._toppest = self._getToppest()
		return self._toppest
	@ property
	def cost(self) -> int:
		if self._hCost == -1:
			solvable = self.calcHeuristic()
			if not solvable:
				self._hCost = 1000000000
		return 4 * self._hCost + len(self.moves)

# preprocessing ----------------------------------
def clipLevel(tiles: Board, targets: list[Pos]) -> tuple[Board, set[Pos]]:
	assert tiles.size
	tiles = np.pad(tiles, 1)
	nonWalls = np.where(tiles != Tiles.WALL)
	topleft = np.min(nonWalls, axis=1) - 1
	tiles = tiles[   topleft[0] : np.max(nonWalls[0]) + 2]
	tiles = tiles[:, topleft[1] : np.max(nonWalls[1]) + 2]
	return tiles.copy(), set([(t - (topleft - 1)) for t in targets])

def _identifyCorners(tiles, targets) -> np.ndarray:
	forbidden = np.zeros_like(tiles, dtype='bool')
	for curr in Pos.iterBoard(tiles):
		if tiles[curr] != Tiles.FREE or curr in targets: continue
		isWallAround = [tiles[curr + move] == Tiles.WALL for move in Moves]
		for a, b in zip(isWallAround, isWallAround[1:] + [isWallAround[0]]):
			if a and b:
				forbidden[curr] = True
				break
	return forbidden
def _extendWall(tiles, targets, corner: Pos, forbidden):
	for dir in [Pos(0, 1), Pos(1, 0)]:
		pos = corner
		foundTiles = []
		started = True
		wallTop, wallBottom = True, True
		while tiles[pos] != Tiles.WALL and pos not in targets and (not forbidden[pos] or started) and (wallTop or wallBottom):
			wallTop = wallTop and tiles[pos - dir] == Tiles.WALL
			wallBottom = wallBottom and tiles[pos + dir] == Tiles.WALL
			foundTiles.append(pos)
			pos += dir[::-1]
			started = False
		if forbidden[pos] and (wallTop or wallBottom):
			for t in foundTiles:
				forbidden[t] = True
def getForbiddenTiles(tiles: Board, targets: set[Pos]) -> Board:
	forbidden = _identifyCorners(tiles, targets)
	corners = forbidden.copy()
	for pos in Pos.iterBoard(corners):
		if corners[pos]:
			_extendWall(tiles, targets, pos, forbidden)
	return forbidden

def _findDistToTarget(state: State, startPos: Pos, target: Pos) -> int:
	# TODO call only for each target
	openedH = Heap((0, target, tuple()))
	closed: set[Pos] = set()
	while openedH:
		dist, opened, prevMoves = openedH.pop()
		if opened == startPos:
			return dist
		closed.add(opened)
		for move in Moves:
			pos = opened + move
			far = pos + move
			if pos in closed or state.tiles[pos] == Tiles.WALL or state.tiles[far] == Tiles.WALL:
				continue
			newDist = dist + 1 + 2 * (move not in prevMoves and len(prevMoves))
			if openedH.hasKey(pos):
				i = openedH.index(pos)
				openedH.changeMetadata(i, (openedH.heap[i][2] + (move,),))
				if openedH.heap[i][0] > newDist:
					openedH.decreasePriority(pos, newDist)
			else:
				openedH.push((newDist, pos, (move, )))
	return float('inf')
def computeDistMaps(state: State):
	State.distMaps = np.ndarray((len(state.targets), *state.tiles.shape), 'float32')
	State.distMaps[:] = float('inf')
	for targetIdx, target in enumerate(state.targets):
		for pos in Pos.iterBoard(state.tiles):
			if state.forbidden[pos] or state.tiles[pos] == Tiles.WALL:
				continue
			dist = _findDistToTarget(state, pos, target)
			State.distMaps[targetIdx][pos] = dist

# solving ---------------------------------------------
def communicateSolving(comms: Comms, state: State, *, endSolving=False):
	"""@return whether solving should end"""
	if comms.stopEvent.is_set():
		return True
	comms.stateQueue.put(state)
	if endSolving or state.levelWon():
		moves = [] if endSolving else state.moves
		comms.movesQueue.put(moves)
		comms.doneEvent.set()
		comms.stateQueue.put(state)
		return True

def solveLevel(state: State, comms: Comms):
	closed: set[State] = set()
	heap = Heap(state)
	while heap:
		state = heap.pop()
		closed.add(state)
		if communicateSolving(comms, state):
			return
		for newState in findPossibleRockMoves(state):
			if newState in closed: continue
			if heap.hasItem(newState):
				i = heap.getItemIdx(newState)
				heap.heap[i].takeBetterFrom(newState)
				heap.changedPriority(i)
			else:
				heap.push(newState)
	communicateSolving(comms, state, endSolving=True)
def pullObjectsFromTiles(tiles: Board, targets: set[Pos]) -> tuple[Pos, set[Pos]]:
	assert np.sum(tiles == Tiles.BULLDOZER) == 1, 'exactly one bulldozer expected'
	bulldozerPos = Pos.getCoordsWhere(tiles == Tiles.BULLDOZER, onlyOne=True)
	tiles[bulldozerPos] = Tiles.FREE

	rocks = Pos.getCoordsWhere(tiles == Tiles.ROCK)
	for rockPos in rocks:
		tiles[rockPos] = Tiles.FREE

	if len(rocks) == len(targets) + 1:
		targets.add(bulldozerPos)
	assert len(rocks) == len(targets), 'number of rocks and targets doesn\'t match'
	return bulldozerPos, set(rocks)
def prepareLevel(tiles: Board, targets: list[Pos]) -> State:
	tiles, targets = clipLevel(tiles, targets)
	bulldozerPos, rocks = pullObjectsFromTiles(tiles, targets)
	
	forbidden = getForbiddenTiles(tiles, targets)
	state = State.globalInit(rocks, bulldozerPos, tiles, targets, forbidden)
	computeDistMaps(state)
	return state

def findPossibleRockMoves(state: State) -> list[State]:
	closed: set[Pos] = set((state.bulldozerPos, ))
	opened: deque[State] = deque((state, ))
	newStates: list[State] = []
	while opened:
		s = opened.popleft()
		for move in Moves:
			currS = s.movedBulldozer(move)
			currPos = currS.bulldozerPos
			if currS.bulldozerPos in currS.rocks:
				newRockPos = currS.bulldozerPos + move
				if newRockPos not in currS.rocks and state.tiles[newRockPos] == Tiles.FREE and not state.forbidden[newRockPos]:
					currS.adjustMovedRock(newRockPos)
					newStates.append(currS)
			elif state.tiles[currPos] == Tiles.FREE and currPos not in closed:
				opened.append(currS)
				closed.add(currPos)
	return newStates
