from enum import IntEnum, auto
from collections import deque

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
BoardState = tuple[tuple[tuple[int, int]], tuple[int, int]]
RockMove = tuple[Board, Pos, list[Moves]]

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
def checkLevel(tiles, targets, bulldozerPos):
	# one bulldozer
	assert 1 == sum([sum([t == Tiles.BULLDOZER for t in row]) for row in tiles])

	# as many rocks as targets
	numRocks = sum([sum([t == Tiles.ROCK for t in row]) for row in tiles])
	assert numRocks in [len(targets), len(targets) + 1]
	if numRocks == len(targets) + 1:
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

# solving ---------------------------------------------
lastWindowWait = 0.0
def solveLevel(tiles: Board, targets: list[Pos], bulldozerPos: Pos, forbidden: list[list[bool]]) -> list[Moves]:
	closed = set((stateDesc(tiles, bulldozerPos), ))
	success, moves = solveRecursion(tiles, targets, bulldozerPos, forbidden, closed)
	if not success:
		raise RuntimeError('Couldn\'t find a solution')
	return moves
def solveRecursion(tiles, targets, bulldozerPos, forbidden: list[list[bool]], closed: set[BoardState]) -> tuple[bool, list[Moves]]:
	global lastWindowWait
	if (t := time.time()) - lastWindowWait > 0.1:
		lastWindowWait = t
		if cv2.waitKey(1) == 'q' or cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:
			exit(0)
			
	for board, pos, moves in findPossibleRockMoves(tiles, bulldozerPos, forbidden):
		if levelComplete(board, targets):
			return (True, moves)
		if (desc := stateDesc(board, pos)) in closed: continue
		closed.add(desc)
		success, postmoves = solveRecursion(board, targets, pos, forbidden, closed)
		if success:
			return (True, moves + postmoves)
	return (False, [])			 

def levelComplete(tiles: Board, targets: list[Pos]) -> bool:
	for pos in targets:
		if tiles[pos[1]][pos[0]] != Tiles.ROCK:
			return False
	return True
def prepareLevel(tiles: Board, targets: list[Pos]) -> tuple[Board, list[Pos], Pos, list[list[bool]]]:
	tiles, targets = clipLevel(tiles, targets)
	bulldozerPos = [[tiles[y].index(Tiles.BULLDOZER), y] for y in range(len(tiles)) if Tiles.BULLDOZER in tiles[y]][0]
	checkLevel(tiles, targets, bulldozerPos)
	
	tiles[bulldozerPos[1]][bulldozerPos[0]] = Tiles.FREE
	return tiles, targets, bulldozerPos, getForbiddenTiles(tiles, targets)

def stateDesc(tiles, bulldozerPos: Pos) -> BoardState:
	toppest = len(tiles[0]), len(tiles)
	closed = set((tuple(bulldozerPos), ))
	opened = deque((bulldozerPos, ))
	while opened:
		curr = opened.popleft()
		if curr[1] < toppest[1] or curr[1] == toppest[1] and curr[0] < toppest[0]:
			toppest = tuple(curr)
		for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
			new = curr[0] + (move // 10) - 1, curr[1] + (move % 10) - 1
			if tiles[new[1]][new[0]] == Tiles.FREE and new not in closed:
				opened.append(list(new))
				closed.add(new)
	assert toppest[0] < len(tiles[0]) and toppest[1] < len(tiles)
	rocks = tuple(sum([[(x, y) for x, tile in enumerate(row) if tile == Tiles.ROCK] for y, row in enumerate(tiles)], start=[]))
	return rocks, toppest

def findPossibleRockMoves(tiles: Board, bulldozerPos: Pos, forbidden: list[list[bool]]) -> list[RockMove]:
	closed: set[tuple[int, int]] = set((tuple(bulldozerPos), ))
	opened: deque[tuple[Pos, list[Moves]]] = deque(((bulldozerPos, []), ))
	rockMoves: list[RockMove] = []
	while opened:
		curr, moves = opened.popleft()
		for move in [Moves.UP, Moves.DOWN, Moves.RIGHT, Moves.LEFT]:
			currPos = curr[0] + (move // 10) - 1, curr[1] + (move % 10) - 1
			if tiles[currPos[1]][currPos[0]] == Tiles.ROCK:
				newRockPos = [currPos[0] + (move // 10) - 1, currPos[1] + (move % 10) - 1]
				if tiles[newRockPos[1]][newRockPos[0]] == Tiles.FREE and not forbidden[newRockPos[1]][newRockPos[0]]:
					newTiles = _copyTiles(tiles)
					newTiles[newRockPos[1]][newRockPos[0]] = Tiles.ROCK
					newTiles[currPos[1]][currPos[0]] = Tiles.FREE
					rockMoves.append([newTiles, list(currPos), moves + [move]])
			elif tiles[currPos[1]][currPos[0]] == Tiles.FREE and currPos not in closed:
				opened.append((list(currPos), moves + [move]))
				closed.add(currPos)
	return rockMoves
