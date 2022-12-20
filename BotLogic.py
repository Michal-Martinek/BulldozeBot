from enum import IntEnum, auto
from collections import deque
import heapq
from dataclasses import dataclass
import cv2, time

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

Pos = list[int]
Board = list[list[Tiles]]
DistMap = list[list[int]]

@ dataclass
class State:
	# TODO: rocks and targets should be a set()
	rocks: list[Pos]
	bulldozerPos: Pos
	moves: list[Moves]
	tiles: list[list[Tiles]]
	targets: list[Pos]
	forbidden: list[list[bool]]
	distMaps: list[DistMap]

	def levelWon(self):
		return all([t in self.rocks for t in self.targets])
	def movedBulldozer(self, move: Moves):
		return State(self.rocks.copy(), self.movedPos(move), self.moves + [move], self.tiles, self.targets, self.forbidden, self.distMaps)
	def movedPos(self, move: Moves) -> Pos:
		return [self.bulldozerPos[0] + (move // 10) - 1, self.bulldozerPos[1] + (move %  10) - 1]
	def adjustMovedRock(self, newRockPos: Pos):
		self.rocks.remove(self.bulldozerPos)
		self.rocks.append(newRockPos)
	def __hash__(self) -> int: # TODO
		return hash(f'{set([tuple(p) for p in self.rocks])}-{self.bulldozerPos}')
	def __lt__(self, other):
		return len(self.moves) < len(other.moves)
	def _getToppest(self) -> Pos:
		toppest = [len(self.tiles[0]), len(self.tiles)]
		closed = set((tuple(self.bulldozerPos), ))
		opened = deque((self.bulldozerPos, ))
		while opened:
			curr = opened.popleft()
			if curr[1] < toppest[1] or curr[1] == toppest[1] and curr[0] < toppest[0]:
				toppest = curr
			for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
				new = curr[0] + (move // 10) - 1, curr[1] + (move % 10) - 1
				if self.tiles[new[1]][new[0]] == Tiles.FREE and new not in closed:
					opened.append(list(new))
					closed.add(new)
		assert toppest[0] < len(self.tiles[0]) and toppest[1] < len(self.tiles)
		return toppest
	def __eq__(self, other):
		return set([tuple(p) for p in self.rocks]) == set([tuple(p) for p in other.rocks]) and self._getToppest() == other._getToppest()

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
	
	return _copyTiles(tiles), [[t[0] + offsets[1], t[1] + offsets[0]] for t in targets]
def checkLevel(tiles, targets, bulldozerPos, rocks):
	# one bulldozer # TODO: test these
	assert 1 == sum([sum([t == Tiles.BULLDOZER for t in row]) for row in tiles])

	# as many rocks as targets
	assert len(rocks) in [len(targets), len(targets) + 1]
	if len(rocks) == len(targets) + 1:
		targets.append(bulldozerPos)
	
	# targets not on walls
	assert all([[tiles[y][x] != Tiles.WALL] for x, y in targets])
	
	# walls around level
	assert all([tiles[0][x] == Tiles.WALL and tiles[-1][x] == Tiles.WALL for x in range(len(tiles[0]))])
	assert all([tiles[y][0] == Tiles.WALL and tiles[y][-1] == Tiles.WALL for y in range(len(tiles))])

def _identifyCorners(tiles, targets) -> list[list[bool]]:
	forbidden = [[False for x in range(len(tiles[0]))] for y in range(len(tiles))]
	for x in range(1, len(tiles[0])-1):
		for y in range(1, len(tiles)-1):
			walls = []
			for i, move in enumerate([Moves.UP, Moves.RIGHT, Moves.DOWN, Moves.LEFT]):
				pos = x + (move // 10) - 1, y + (move % 10) - 1
				if tiles[pos[1]][pos[0]] == Tiles.WALL:
					walls.append(i)
			isCorner = any([(i in walls) and ((i+1)%4 in walls) for i in range(4)])
			if [x, y] not in targets and isCorner and tiles[y][x] == Tiles.FREE:
				forbidden[y][x] = True
	return forbidden
def _extendWall(tiles, targets, x, y, forbidden):
	# assert False # TODO: test ---| junction
	savedX, savedY = x, y
	for xOff, yOff in [(0, 1), (1, 0)]:
		x, y = savedX, savedY
		foundTiles = []
		started = True
		wallTop, wallBottom = True, True
		while tiles[y][x] != Tiles.WALL and [x, y] not in targets and (not forbidden[y][x] or started) and (wallTop or wallBottom):
			wallTop = wallTop and tiles[y-xOff][x-yOff] == Tiles.WALL
			wallBottom = wallBottom and tiles[y+xOff][x+yOff] == Tiles.WALL
			foundTiles.append([x, y])
			x += xOff
			y += yOff
			started = False
		if forbidden[y][x] and (wallTop or wallBottom):
			for tx, ty in foundTiles:
				forbidden[ty][tx] = True
def getForbiddenTiles(tiles, targets) -> list[list[bool]]:
	forbidden = _identifyCorners(tiles, targets)
	corners = _copyTiles(forbidden)
	for x in range(1, len(tiles[0])-1):
		for y in range(1, len(tiles)-1):
			if corners[y][x]:
				_extendWall(tiles, targets, x, y, forbidden)
	return forbidden
def _copyTiles(tiles: Board):
	return [[t for t in row] for row in tiles]

def _findDistToTarget(state: State, startPos: Pos, target: Pos) -> int:
	openedH: list[tuple[int, Pos, set[Moves]]] = [(0, target, set())]
	openedD: dict[tuple[int, int]] = {tuple(target): 0}
	closed: set[tuple[int, int]] = set()
	while openedH:
		dist, opened, prevMoves = heapq.heappop(openedH)
		if opened == startPos:
			return dist
		closed.add(tuple(opened))
		openedD.pop(tuple(opened))
		for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
			pos = [opened[0] + (move // 10) - 1, opened[1] + (move %  10) - 1]
			far = [pos[0] + (move // 10) - 1, pos[1] + (move %  10) - 1]
			if tuple(pos) in closed or state.tiles[pos[1]][pos[0]] == Tiles.WALL or state.tiles[far[1]][far[0]] == Tiles.WALL:
				continue
			newDist = dist + 1 + 2 * (move not in prevMoves and len(prevMoves)) # TODO: if its around corner, we should check the len of the path
			if tuple(pos) in openedD:
				for i, entry in enumerate(openedH):
					if entry[1] == pos:
						entry[2].add(move)

				if openedD[tuple(pos)] > newDist:
					for i, entry in enumerate(openedH):
						if entry[1] == pos:
							openedH.pop(i)
							break
					else: assert False
					openedH.append((newDist, pos, entry[2]))
					openedD[tuple(pos)] = newDist
					heapq.heapify(openedH)
			else:
				heapq.heappush(openedH, (newDist, pos, set((move, ))))
				openedD[tuple(pos)] = newDist
	return -1
def computeDistMaps(state: State):
	for target in state.targets:
		dists = [[-1 for x in range(len(state.tiles[0]))] for y in range(len(state.tiles))]
		for y in range(len(state.tiles)):
			for x in range(len(state.tiles[0])):
				if state.forbidden[y][x] or state.tiles[y][x] == Tiles.WALL:
					continue
				dist = _findDistToTarget(state, [x, y], target)
				dists[y][x] = dist
		state.distMaps.append(dists)

# solving ---------------------------------------------
lastWindowWait = 0.0
def solveLevel(startState) -> list[Moves]:
	closed = set((startState, ))
	heap: list[State] = []
	heapq.heappush(heap, startState)
	outMoves = []
	lastWindowWait = 0.0
	while len(heap):
		if (t := time.time()) - lastWindowWait > 0.1:
			lastWindowWait = t
			if cv2.waitKey(1) == 'q' or cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:
				exit(0)

		state = heapq.heappop(heap)
		if state.levelWon():
				if len(state.moves) < len(outMoves) or outMoves == []:
					outMoves = state.moves
				continue
		for newState in findPossibleRockMoves(state):
			if newState in closed: continue
			closed.add(newState)
			heapq.heappush(heap, newState)
	return outMoves

def prepareLevel(tiles: Board, targets: list[Pos]) -> State:
	tiles, targets = clipLevel(tiles, targets)
	bulldozerPos = [[tiles[y].index(Tiles.BULLDOZER), y] for y in range(len(tiles)) if Tiles.BULLDOZER in tiles[y]][0]
	rocks = sum([[[x, y] for x, tile in enumerate(row) if tile == Tiles.ROCK] for y, row in enumerate(tiles)], start=[])
	
	checkLevel(tiles, targets, bulldozerPos, rocks)
	
	tiles[bulldozerPos[1]][bulldozerPos[0]] = Tiles.FREE
	forbidden = getForbiddenTiles(tiles, targets)
	state = State(rocks, bulldozerPos, [], tiles, targets, forbidden, [])
	computeDistMaps(state)
	for rX, rY in rocks:
		state.tiles[rY][rX] = Tiles.FREE
	assert len(state.distMaps) == len(state.targets)
	return state

def findPossibleRockMoves(state: State) -> list[State]:
	closed: set[tuple[int, int]] = set((tuple(state.bulldozerPos), ))
	opened: deque[State] = deque((state, ))
	newStates: list[State] = []
	while opened:
		s = opened.popleft()
		for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
			currS = s.movedBulldozer(move)
			currPos = tuple(currS.bulldozerPos)
			if currS.bulldozerPos in currS.rocks:
				newRockPos = currS.movedPos(move)
				if newRockPos not in currS.rocks and state.tiles[newRockPos[1]][newRockPos[0]] == Tiles.FREE and not state.forbidden[newRockPos[1]][newRockPos[0]]:
					currS.adjustMovedRock(newRockPos)
					newStates.append(currS)
			elif state.tiles[currPos[1]][currPos[0]] == Tiles.FREE and currPos not in closed:
				opened.append(currS)
				closed.add(currPos)
	return newStates
