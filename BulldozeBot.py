import win32gui, win32ui, win32con
import numpy as np
import cv2
import re, os
import subprocess

import BotLogic
from BotLogic import State
from Classes import Tiles, Moves, Pos

os.chdir(os.path.dirname(__file__))

# window capture ---------------------------------
def startBuldozeExe():
	print('INFO: starting BULLDOZE.exe')
	subprocess.Popen('"BULLDOZE Game\\BULLDOZE.exe"', shell=False, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
	cv2.waitKey(500)
def getPossibleWindows(errorOnNone=False) -> dict[int, str]:
	windows = {}
	windowRegex = 'Bulldozer - Level \\d+'
	def winEnumHandler(hwnd, ctx):
		if not win32gui.IsWindowVisible(hwnd): return
		name = win32gui.GetWindowText(hwnd)
		if re.match(windowRegex, name):
			windows[hwnd] = name
	
	win32gui.EnumWindows(winEnumHandler, None)
	assert not errorOnNone or len(windows), 'Could not find Bulldozer game window'
	return windows
def findBulldozerWindow() -> int:
	potentialWindows = getPossibleWindows()
	if not len(potentialWindows):
		startBuldozeExe()
		potentialWindows = getPossibleWindows(True)
	if len(potentialWindows) > 1:
		print('WARNING: Multiple Buldoze game windows found')
	hwnd = tuple(potentialWindows.keys())[0]
	print(f'Found window "{win32gui.GetWindowText(hwnd)}"')
	return hwnd

def getScreenshot(hwnd, cropped_x=8, cropped_y=30):
		# maximize the window
		win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
		cv2.waitKey(1)

		LUX, LUY, RBX, RBY = win32gui.GetWindowRect(hwnd)
		W, H = RBX - LUX, RBY - LUY
		# get the window image data
		wDC = win32gui.GetWindowDC(hwnd)
		dcObj = win32ui.CreateDCFromHandle(wDC)
		cDC = dcObj.CreateCompatibleDC()
		dataBitMap = win32ui.CreateBitmap()
		dataBitMap.CreateCompatibleBitmap(dcObj, W, H)
		cDC.SelectObject(dataBitMap)
		cDC.BitBlt((0, 0), (W, H), dcObj, (cropped_x, cropped_y), win32con.SRCCOPY)

		# convert the raw data into a format opencv can read
		# dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
		signedIntsArray = dataBitMap.GetBitmapBits(True)
		img = np.frombuffer(signedIntsArray, dtype='uint8')
		img.shape = (H, W, 4)
		img = img[...,:3]

		# free resources
		dcObj.DeleteDC()
		cDC.DeleteDC()
		win32gui.ReleaseDC(hwnd, wDC)
		win32gui.DeleteObject(dataBitMap.GetHandle())

		# make image C_CONTIGUOUS to avoid errors that look like:
		#   File ... in draw_rectangles
		#   TypeError: an integer is required (got type tuple)
		# see the discussion here:
		# https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
		img = np.ascontiguousarray(img)
		return img

# templates --------------------------------------
TILESIZE = 32
templates = {
	file.split('.')[0] : cv2.imread(os.path.join('Templates', file), cv2.IMREAD_COLOR)
		for file in os.listdir('Templates')
}
# drawing -----------------------------------
def blit(img, src, x_offset: int, y_offset: int):
	img[y_offset:y_offset+src.shape[0], x_offset:x_offset+src.shape[1]] = src
def drawHollowRect(img, color, x, y, w, h, thickness=2):
	img[y:y+thickness, x:x+w] = color
	img[y+h-thickness:y+h, x:x+w] = color
	img[y:y+h, x:x+thickness] = color
	img[y:y+h, x+w-thickness:x+w] = color
def boxLocations(img, locs, templShape, color):
	for y, x in zip(*locs):
			cv2.rectangle(img, (x, y), (x + templShape[0], y + templShape[1]), color, 2, cv2.LINE_4)
def drawGridlines(img):
	img[0:img.shape[0]:TILESIZE] = (0, 0, 0)
	img[:, 0:img.shape[1]:TILESIZE] = (0, 0, 0)
def _getImg(tile: Tiles, isTarget: bool, forbidden: bool):
	tileName = {Tiles.FREE: ['Free', 'Free-forbidden'][forbidden],	Tiles.BULLDOZER: ['Bulldozer-up', 'Bulldozer-forbidden'][forbidden], Tiles.ROCK: 'Rock', Tiles.WALL: 'Wall'}[tile]
	if isTarget:
		if tile == Tiles.BULLDOZER:
			tileName = 'Bulldozer-target'
		elif tile == Tiles.ROCK:
			tileName = 'RockOnTarget'
		else: tileName = 'Target'
	return templates[tileName]
def drawDetectedLevel(state: State):
	img = np.zeros((len(state.tiles) * TILESIZE, len(state.tiles[0]) * TILESIZE, 3), dtype='uint8')
	for pos in Pos.iterBoard(state.tiles, inner=False):
		tile = state.tiles[pos.y][pos.x]
		if pos == state.bulldozerPos:
			tile = Tiles.BULLDOZER
		if pos in state.rocks:
			tile = Tiles.ROCK
		template = _getImg(tile, pos in state.targets, state.forbidden[pos.y][pos.x])
		blit(img, template, pos.x * TILESIZE, pos.y * TILESIZE)
	return img
# object detection --------------------------------------
def clipScreenshot(img):
	matched = cv2.matchTemplate(img, templates['Rock'], cv2.TM_CCOEFF_NORMED)
	posses = np.array( np.where(matched > 0.90) )
	assert posses.size, 'Could not find any rocks, the level is probably solved'
	minPos = np.min(posses, axis=1) % TILESIZE
	img = img[minPos[0]:, minPos[1]:]
	height, width = img.shape[0] // TILESIZE, img.shape[1] // TILESIZE
	return img[:height * TILESIZE, :width * TILESIZE].copy(), (height, width)
def getBestMatch(img) -> str:
	bestMatch = 'Free'
	bestVal = 0.0
	names = ['Bulldozer-down', 'Bulldozer-left', 'Bulldozer-right', 'Bulldozer-up', 'Rock', 'RockOnTarget', 'Target', 'Wall', 'Wall2', 'Wall3', 'Wall4']
	for name in names:
		tile = templates[name]
		assert tile.shape == (TILESIZE, TILESIZE, 3)
		val = cv2.matchTemplate(img, tile, cv2.TM_CCOEFF_NORMED)
		assert val.shape == (1, 1)
		val = val[0, 0]
		if val > bestVal:
			bestMatch = name
			bestVal = val
	return bestMatch
def matchBlock(img) -> tuple[Tiles, bool]:
	assert img.shape == (TILESIZE, TILESIZE, 3)
	matchName = getBestMatch(img)
	matched = Tiles.FREE
	if matchName in ['Bulldozer-down', 'Bulldozer-left', 'Bulldozer-right', 'Bulldozer-up']:
		matched = Tiles.BULLDOZER
	elif matchName in ['Rock', 'RockOnTarget']:
		matched = Tiles.ROCK
	elif matchName in ['Wall', 'Wall2', 'Wall3', 'Wall4']:
		matched = Tiles.WALL
	return matched, matchName in ['RockOnTarget', 'Target']
	
def detectLevel(img: np.ndarray, height, width) -> tuple[list[list[Tiles]], list[Pos]]:
	blocks = img.reshape((height, TILESIZE, width, TILESIZE, 3))
	tiles, targets = [], []
	for y in range(height):
		tiles.append([])
		for x in range(width):
			tile, target = matchBlock(blocks[y, :, x])
			tiles[-1].append(tile)
			if target:
				targets.append(Pos(x, y))
	return tiles, targets

# executing moves ------------------------------------
def executeMoves(moves: list[Moves], hwnd, duration=20):
	mapp = {Moves.UP: 'W', Moves.DOWN: 'S', Moves.RIGHT: 'D', Moves.LEFT: 'A'}
	for move in [ord(mapp[m]) for m in moves]:
		win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, move, 0)
		if cv2.waitKey(duration) == ord('q') or cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:
			return
		win32gui.SendMessage(hwnd, win32con.WM_KEYUP, move, 0)

# run
def main():
	hwnd = findBulldozerWindow()
	img = getScreenshot(hwnd)
	
	img, dims = clipScreenshot(img)
	tiles, targets = detectLevel(img, *dims)
	state = BotLogic.prepareLevel(tiles, targets)

	img = drawDetectedLevel(state)
	cv2.imshow('BulldozeBot', img)
	cv2.waitKey(1)
	
	moves = BotLogic.solveLevel(state)
	print(f'INFO: found a solution with {len(moves)} moves')
	executeMoves(moves, hwnd)

	retry = False
	while True:
		if cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:
			break
		if (key := cv2.waitKey(1)) in [ord('q'), ord('r')]:
			if key == ord('r'):
				retry = True
			break

	cv2.destroyAllWindows()
	return retry

if __name__ == '__main__':
	while main():
		pass
