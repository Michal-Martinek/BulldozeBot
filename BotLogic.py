from collections import deque
import numpy as np
from Classes import *

COST_COEFF_pairingEstimate = 4.
COST_COEFF_chamberSize = 0.25

class State:
	tiles: np.ndarray
	targets: set[Pos]
	distMaps: np.ndarray=None
	
	@classmethod
	def globalInit(cls, rocks: set[Pos], bulldozerPos: Pos, tiles: Board, targets: set[Pos]):
		cls.tiles = tiles
		cls.targets = targets
		cls.computeDistMaps()
		return cls(rocks, bulldozerPos)

	def __init__(self, rocks: set[Pos], bulldozerPos: Pos, moves: list[Moves]=[]):
		self.rocks: set[Pos] = rocks
		self.bulldozerPos = bulldozerPos
		self.moves: list[Moves] = moves
		# costs
		self.pCost = len(moves)
		self.pairingEstimate = self.calcPairingEstimate()
		# TODO cost for child states being unsolvable
		# NOTE chamber = space accesible without moving any rocks
		self.chamberTop: Pos = None
		self.chamberSize: int = None
		self.rockMoves: list[tuple[Pos, list[Moves]]] = []
		self.exploreChamber()
	@classmethod
	def afterRockMove(cls, parentState, moves: list[Moves], rockPos: Pos, newRockPos: Pos):
		newRocks: set[Pos] = parentState.rocks.copy()
		newRocks.remove(rockPos)
		newRocks.add(newRockPos)
		return State(newRocks, rockPos, parentState.moves + moves)
	@classmethod
	def computeDistMaps(cls):
		cls.distMaps = np.ndarray((len(cls.targets), *cls.tiles.shape), 'float32')
		cls.distMaps[:] = float('inf')
		for targetIdx, target in enumerate(cls.targets):
			for pos in Pos.iterBoard(cls.tiles):
				if cls.tiles[pos] == Tiles.WALL:
					continue
				dist = findDistToTarget(cls.tiles, pos, target)
				cls.distMaps[targetIdx][pos] = dist
	def __getitem__(self, pos: Pos):
		return self.tiles[pos]
	def isForbidden(self, pos: Pos):
		return bool( np.all(self.distMaps[:, *pos] == float('inf')) )
	def isAt(self, pos, tile: Tiles, checkRock=False):
		return self.tiles[pos] == tile and not (checkRock and pos in self.rocks)
	def levelWon(self) -> bool:
		return self.rocks == self.targets
	@property
	def cost(self) -> int:
		cost = self.pCost
		cost += COST_COEFF_pairingEstimate * self.pairingEstimate
		cost -= COST_COEFF_chamberSize * self.chamberSize # TODO TEST
		return cost

	def takeBetterFrom(self, other):
		if len(self.moves) > len(other.moves):
			# findDistToTarget(self.tiles, self.bulldozerPos, )
			# NOTE: update only if states have same bulldozer pos
			if self.bulldozerPos == other.bulldozerPos:
				self.moves = other.moves.copy()

	def __hash__(self) -> int:
		return hash((tuple(self.rocks), self.chamberTop))
	def __repr__(self) -> str:
		c = self.cost
		return f'State({self.bulldozerPos}, c={c}, h={self.pairingEstimate}, rocks={self.rocks})'
	def __lt__(self, other):
		if self.cost == other.cost:
			return self.pairingEstimate < other.pairingEstimate
		return self.cost < other.cost
	def __eq__(self, other):
		return self.rocks == other.rocks and self.chamberTop == other.chamberTop

	def exploreChamber(self) -> Pos:
		chamberTop = Pos(self.tiles.shape)
		closed = set((self.bulldozerPos, ))
		opened = deque(((self.bulldozerPos, []), ))
		self.chamberSize = 0
		while opened:
			pos, moves = opened.popleft()
			self.chamberSize += 1
			if pos < chamberTop:
				chamberTop = pos
			for move in Moves:
				newPos = pos + move
				if self.tiles[newPos] != Tiles.FREE or newPos in closed:
					continue
				newState = newPos, moves + [move]
				if newPos in self.rocks:
					self.rockMoves.append(newState)
				else:
					opened.append(newState)
					closed.add(newPos)
		assert self.tiles[chamberTop] == Tiles.FREE
		self.chamberTop = chamberTop
	def genPossibleRockMoves(self):
		for rockPos, moves in self.rockMoves:
			newRockPos = rockPos + moves[-1]
			if newRockPos not in self.rocks and self.tiles[newRockPos] == Tiles.FREE:
				state = State.afterRockMove(self, moves, rockPos, newRockPos)
				if state.hasBlockedRocks(newRockPos):
					continue
				yield state
	def hasBlockedRocks(self, rockPos: Pos) -> bool:
		if rockPos in self.targets: return False
		for neighborRock in Pos.iter3by3(rockPos, crossOnly=True):
			if neighborRock not in self.rocks: continue
			orthogonalMove = Pos((rockPos - neighborRock)[::-1])
			for sideStep in (orthogonalMove, orthogonalMove * -1):
				posesAlong = rockPos + sideStep, neighborRock + sideStep
				blockedAlong = [(self.tiles[a] == Tiles.WALL or a in self.rocks) for a in posesAlong]
				if all(blockedAlong): return True

	# heuristics --------------------------------------------
	def calcPairingEstimate(self) -> float:
		'''calculates heuristic by deciding final target for each rock
		* returns somewhat optimal such pairing, 'inf' if unsolvable'''
		table = np.zeros((len(self.rocks), len(self.targets)), dtype='float32')
		for r, rockPos in enumerate(self.rocks):
			for m, distMap in enumerate(self.distMaps):
				cost = distMap[rockPos]
				table[r, m] = cost
		# sort table's rows with most infs up top to prevent backtracking
		numInfs = np.sum(table == float('inf'), axis=1)
		table = table[np.argsort(-numInfs)]
		return self._pairRecusively(table)
	def _pairRecusively(self, table: np.ndarray):
		if table.size == 4:
			return min(table[0, 0] + table[1, 1], table[0, 1] + table[1, 0])
		if np.max(np.sum(table == float('inf'), axis=0)) == table.shape[0]:
			return float('inf') # exists col with infs only
		for col in np.argsort(table[0]):
			if table[0, col] == float('inf'): break # there are only infs left in this row
			otherColIdxs = np.delete(np.arange(table.shape[1]), col)
			cost = self._pairRecusively(table[1:, otherColIdxs])
			if cost == float('inf'): continue
			return cost + table[0][col]
		return float('inf')

# preprocessing ----------------------------------
def clipLevel(tiles: Board, targets: list[Pos]) -> tuple[Board, set[Pos]]:
	assert tiles.size
	tiles = np.pad(tiles, 1)
	nonWalls = np.where(tiles != Tiles.WALL)
	topleft = np.min(nonWalls, axis=1) - 1
	tiles = tiles[   topleft[0] : np.max(nonWalls[0]) + 2]
	tiles = tiles[:, topleft[1] : np.max(nonWalls[1]) + 2]
	return tiles.copy(), set([(t - (topleft - 1)) for t in targets])
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
	return State.globalInit(rocks, bulldozerPos, tiles, targets)

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
		for newState in state.genPossibleRockMoves():
			if newState in closed or newState.cost == float('inf'):
				continue
			if heap.hasItem(newState):
				i = heap.getItemIdx(newState)
				heap.heap[i].takeBetterFrom(newState)
				heap.changedPriority(i)
			else:
				heap.push(newState)
	communicateSolving(comms, state, endSolving=True)
def findDistToTarget(tiles: Board, startPos: Pos, target: Pos) -> int:
	'''min required moves to move rock from to target'''
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
			if pos in closed or tiles[pos] == Tiles.WALL or tiles[far] == Tiles.WALL:
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
