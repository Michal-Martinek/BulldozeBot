from enum import IntEnum, auto

class Tiles(IntEnum):
	WALL = auto()
	FREE = auto()
	ROCK = auto()
	BULLDOZER = auto()

# preprocessing ----------------------------------
def clipLevel(tiles: list[list[Tiles]], targets: list[tuple[int]]) -> tuple[list[list[Tiles]], list[tuple[int]]]:
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
	
	return tiles, [(t[0] + offsets[1], t[1] + offsets[0]) for t in targets]
def checkLevel(tiles, targets):
	# targets on free or rocks
	assert all([tiles[y][x] in [Tiles.FREE, Tiles.ROCK] for x, y in targets])
	# walls around level
	assert all([tiles[0][x] == Tiles.WALL and tiles[-1][x] == Tiles.WALL for x in range(len(tiles[0]))])
	assert all([tiles[y][0] == Tiles.WALL and tiles[y][-1] == Tiles.WALL for y in range(len(tiles))])
	# as many rocks as targets
	assert len(targets) == sum([sum([t == Tiles.ROCK for t in row]) for row in tiles])
	# one bulldozer
	assert 1 == sum([sum([t == Tiles.BULLDOZER for t in row]) for row in tiles])
