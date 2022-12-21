from collections import deque
import cv2, time
from Classes import Tiles, Moves, Pos, Heap

Board = list[list[Tiles]]
DistMap = list[list[int]]

class State: # TODO: add costs to states
	def __init__(self, rocks: set[Pos], bulldozerPos: Pos, moves: list[Moves], tiles: list[list[Tiles]], targets: set[Pos], forbidden: list[list[bool]], distMaps: list[DistMap]):
		self.rocks = rocks
		self.bulldozerPos = bulldozerPos
		self.moves = moves
		self.tiles = tiles
		self.targets = targets
		self.forbidden = forbidden
		self.distMaps = distMaps
	def levelWon(self):
		return all([t in self.rocks for t in self.targets])
	def movedBulldozer(self, move: Moves):
		return State(self.rocks.copy(), self.bulldozerPos.moved(move), self.moves + [move], self.tiles, self.targets, self.forbidden, self.distMaps)
	def adjustMovedRock(self, newRockPos: Pos):
		self.rocks.remove(self.bulldozerPos)
		self.rocks.add(newRockPos)
	def __hash__(self) -> int: # TODO
		return hash(f'{set([tuple(p) for p in self.rocks])}-{self.bulldozerPos}')
	def __lt__(self, other):
		return len(self.moves) < len(other.moves)
	def _getToppest(self) -> Pos:
		toppest = Pos(len(self.tiles[0]), len(self.tiles))
		closed = set((self.bulldozerPos, ))
		opened = deque((self.bulldozerPos, ))
		while opened:
			curr = opened.popleft()
			if curr.y < toppest.y or curr.y == toppest.y and curr.x < toppest.x:
				toppest = curr
			for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
				new = curr.moved(move)
				if new.at(self.tiles)== Tiles.FREE and new not in closed:
					opened.append(new)
					closed.add(new)
		assert toppest.x < len(self.tiles[0]) and toppest.y < len(self.tiles)
		return toppest
	def __eq__(self, other):
		return self.rocks == other.rocks and self._getToppest() == other._getToppest()

# preprocessing ----------------------------------
def clipLevel(tiles: Board, targets: list[Pos]) -> tuple[Board, list[Pos]]:
	assert len(tiles) > 0 and len(tiles[0]) > 0
	offsets = [0, 0]
	wallRows = [all([t == Tiles.WALL for t in row]) for row in tiles]
	wallRow = [Tiles.WALL for _ in range(len(tiles[0]))]
	if wallRows[0] is False: # top
		tiles.insert(0, wallRow)
		offsets[0] = 1
	else:
		first = wallRows.index(False)
		tiles = tiles[first-1:]
		offsets[0] = 1 - first
	
	if wallRows[-1] is False: # down
		tiles.append(wallRow)
	else:
		last = wallRows[::-1].index(False)
		if last > 1:
			tiles = tiles[:1-last]
	
	wallCols = [all([tiles[y][x] == Tiles.WALL for y in range(len(tiles))]) for x in range(len(tiles[0]))]
	if wallCols[0] is False: # left
		tiles = [[Tiles.WALL] + row for row in tiles]
		offsets[1] = 1
	else:
		first = wallCols.index(False)
		tiles = [row[first-1:] for row in tiles]
		offsets[1] = 1 - first

	if wallCols[-1] is False: # right
		tiles = [row + [Tiles.WALL] for row in tiles]
	else:
		last = wallCols[::-1].index(False)
		if last > 1:
			tiles = [row[:1-last] for row in tiles]
	
	return _copyTiles(tiles), set([Pos(t.x + offsets[1], t.y + offsets[0]) for t in targets])
def checkLevel(tiles, targets, bulldozerPos, rocks):
	# one bulldozer # TODO: test these
	assert 1 == sum([sum([t == Tiles.BULLDOZER for t in row]) for row in tiles])

	# as many rocks as targets
	assert len(rocks) in [len(targets), len(targets) + 1]
	if len(rocks) == len(targets) + 1:
		targets.add(bulldozerPos)
	
	# targets not on walls
	assert all([[pos.at(tiles) != Tiles.WALL] for pos in targets])
	
	# walls around level
	assert all([tiles[0][x] == Tiles.WALL and tiles[-1][x] == Tiles.WALL for x in range(len(tiles[0]))])
	assert all([tiles[y][0] == Tiles.WALL and tiles[y][-1] == Tiles.WALL for y in range(len(tiles))])

def _identifyCorners(tiles, targets) -> list[list[bool]]:
	forbidden = [[False for x in range(len(tiles[0]))] for y in range(len(tiles))]
	for curr in Pos.iterBoard(tiles):
		walls = []
		for i, move in enumerate([Moves.UP, Moves.RIGHT, Moves.DOWN, Moves.LEFT]):
			pos = curr.moved(move)
			if pos.at(tiles) == Tiles.WALL:
				walls.append(i)
		isCorner = any([(i in walls) and ((i+1)%4 in walls) for i in range(4)])
		if curr not in targets and isCorner and curr.at(tiles) == Tiles.FREE:
			forbidden[curr.y][curr.x] = True
	return forbidden
def _extendWall(tiles, targets, pos: Pos, forbidden):
	saved = pos
	for xOff, yOff in [(0, 1), (1, 0)]:
		pos = saved.copy()
		foundTiles = []
		started = True
		wallTop, wallBottom = True, True
		while pos.at(tiles) != Tiles.WALL and pos not in targets and (not pos.at(forbidden) or started) and (wallTop or wallBottom):
			wallTop = wallTop and tiles[pos.y-xOff][pos.x-yOff] == Tiles.WALL
			wallBottom = wallBottom and tiles[pos.y+xOff][pos.x+yOff] == Tiles.WALL
			foundTiles.append(pos.copy())
			pos.x += xOff
			pos.y += yOff
			started = False
		if pos.at(forbidden) and (wallTop or wallBottom):
			for tx, ty in foundTiles:
				forbidden[ty][tx] = True
def getForbiddenTiles(tiles, targets) -> list[list[bool]]:
	forbidden = _identifyCorners(tiles, targets)
	corners = _copyTiles(forbidden)
	for pos in Pos.iterBoard(tiles):
		if pos.at(corners):
			_extendWall(tiles, targets, pos, forbidden)
	return forbidden
def _copyTiles(tiles: Board):
	return [[t for t in row] for row in tiles]

def _findDistToTarget(state: State, startPos: Pos, target: Pos) -> int:
	openedH = Heap((0, target, set()))
	closed: set[Pos] = set()
	while openedH:
		dist, opened, prevMoves = openedH.pop()
		if opened == startPos:
			return dist
		closed.add(opened)
		for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
			pos = opened.moved(move)
			far = pos.moved(move)
			if pos in closed or pos.at(state.tiles) == Tiles.WALL or far.at(state.tiles) == Tiles.WALL:
				continue
			newDist = dist + 1 + 2 * (move not in prevMoves and len(prevMoves)) # TODO: if its around corner, we should check the len of the path
			if openedH.has(pos):
				openedH[pos][2].add(move)
				if openedH[pos][0] > newDist:
					openedH.decreasePriority(pos, newDist)
			else:
				openedH.push((newDist, pos, set((move, ))))
	return -1
def computeDistMaps(state: State):
	for target in state.targets:
		dists = [[-1 for x in range(len(state.tiles[0]))] for y in range(len(state.tiles))]
		for pos in Pos.iterBoard(state.tiles):
			if state.forbidden[pos.y][pos.x] or state.tiles[pos.y][pos.x] == Tiles.WALL:
				continue
			dist = _findDistToTarget(state, pos, target)
			dists[pos.y][pos.x] = dist
		state.distMaps.append(dists)

# solving ---------------------------------------------
lastWindowWait = 0.0
def solveLevel(startState) -> list[Moves]:
	closed = set((startState, ))
	heap = Heap((0, startState))
	outMoves = []
	lastWindowWait = 0.0
	while heap:
		if (t := time.time()) - lastWindowWait > 0.1:
			lastWindowWait = t
			if cv2.waitKey(1) == 'q' or cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:
				exit(0)

		state: State = heap.pop()[1]
		if state.levelWon():
			if len(state.moves) < len(outMoves) or outMoves == []:
				outMoves = state.moves
			continue
		for newState in findPossibleRockMoves(state):
			if newState in closed: continue
			closed.add(newState)
			heap.push((0, newState))
	return outMoves

def prepareLevel(tiles: Board, targets: list[Pos]) -> State:
	tiles, targets = clipLevel(tiles, targets)
	bulldozerPos = Pos(*[[tiles[y].index(Tiles.BULLDOZER), y] for y in range(len(tiles)) if Tiles.BULLDOZER in tiles[y]][0])
	rocks = set(sum([[Pos(x, y) for x, tile in enumerate(row) if tile == Tiles.ROCK] for y, row in enumerate(tiles)], start=[]))
	
	checkLevel(tiles, targets, bulldozerPos, rocks)
	
	tiles[bulldozerPos.y][bulldozerPos.x] = Tiles.FREE
	forbidden = getForbiddenTiles(tiles, targets)
	state = State(rocks, bulldozerPos, [], tiles, targets, forbidden, [])
	computeDistMaps(state)
	for rx, ry in rocks:
		state.tiles[ry][rx] = Tiles.FREE
	assert len(state.distMaps) == len(state.targets)
	return state

def findPossibleRockMoves(state: State) -> list[State]:
	closed: set[Pos] = set((state.bulldozerPos, ))
	opened: deque[State] = deque((state, ))
	newStates: list[State] = []
	while opened:
		s = opened.popleft()
		for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
			currS = s.movedBulldozer(move)
			currPos = currS.bulldozerPos
			if currS.bulldozerPos in currS.rocks:
				newRockPos = currS.bulldozerPos.moved(move)
				if newRockPos not in currS.rocks and newRockPos.at(state.tiles) == Tiles.FREE and not newRockPos.at(state.forbidden):
					currS.adjustMovedRock(newRockPos)
					newStates.append(currS)
			elif currPos.at(state.tiles) == Tiles.FREE and currPos not in closed:
				opened.append(currS)
				closed.add(currPos)
	return newStates
