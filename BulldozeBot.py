import win32gui, win32ui, win32con
import time
import numpy as np
import cv2
import re, os

import BotLogic
from BotLogic import Tiles, Moves

# window capture ---------------------------------
windowNames = {}
def winEnumHandler(hwnd, ctx):
	if win32gui.IsWindowVisible(hwnd):
		name = win32gui.GetWindowText(hwnd)
		windowNames[name] = hwnd
def findBulldozerWindow() -> int:
	win32gui.EnumWindows(winEnumHandler, None)
	bulldozerWindowName = 'Bulldozer - Level \\d+'
	potentialWindowNames = [s for s in windowNames if re.match(bulldozerWindowName, s)]
	if len(potentialWindowNames) > 1:
		print('WARNING: Multiple Buldoze game windows found')
	elif len(potentialWindowNames) == 0:
		raise RuntimeError('Could not find Bulldozer game window, windows found: ', windowNames.keys())
	hwnd = windowNames[potentialWindowNames[0]]
	print(f'Found window {win32gui.GetWindowText(hwnd)}')
	return hwnd
	

def getScreenshot(hwnd, cropped_x=8, cropped_y=30):
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
		# img = np.fromstring(signedIntsArray, dtype='uint8')
		img.shape = (H, W, 4)

		# free resources
		dcObj.DeleteDC()
		cDC.DeleteDC()
		win32gui.ReleaseDC(hwnd, wDC)
		win32gui.DeleteObject(dataBitMap.GetHandle())

		# drop the alpha channel, or cv.matchTemplate() will throw an error like:
		#   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
		#   && _img.dims() <= 2 in function 'cv::matchTemplate'
		img = img[...,:3]

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
	'Free': cv2.imread(os.path.join('Templates', 'Free.png'), cv2.IMREAD_COLOR),
	'Bulldozer-down': cv2.imread(os.path.join('Templates', 'Bulldozer-down.png'), cv2.IMREAD_COLOR),
	'Bulldozer-left': cv2.imread(os.path.join('Templates', 'Bulldozer-left.png'), cv2.IMREAD_COLOR),
	'Bulldozer-right': cv2.imread(os.path.join('Templates', 'Bulldozer-right.png'), cv2.IMREAD_COLOR),
	'Bulldozer-up': cv2.imread(os.path.join('Templates', 'Bulldozer-up.png'), cv2.IMREAD_COLOR),
	'Rock': cv2.imread(os.path.join('Templates', 'Rock.png'), cv2.IMREAD_COLOR),
	'RockOnTarget': cv2.imread(os.path.join('Templates', 'RockOnTarget.png'), cv2.IMREAD_COLOR),
	'Target': cv2.imread(os.path.join('Templates', 'Target.png'), cv2.IMREAD_COLOR),
	'Wall':  cv2.imread(os.path.join('Templates', 'Wall.png' ), cv2.IMREAD_COLOR),
	'Wall2': cv2.imread(os.path.join('Templates', 'Wall2.png'), cv2.IMREAD_COLOR),
	'Wall3': cv2.imread(os.path.join('Templates', 'Wall3.png'), cv2.IMREAD_COLOR),
	'Wall4': cv2.imread(os.path.join('Templates', 'Wall4.png'), cv2.IMREAD_COLOR),
}
# drawing -----------------------------------
def blit(img, src, x_offset: int, y_offset: int):
	img[y_offset:y_offset+src.shape[0], x_offset:x_offset+src.shape[1]] = src
def boxLocations(img, locs, templShape, color):
	for y, x in zip(*locs):
			cv2.rectangle(img, (x, y), (x + templShape[0], y + templShape[1]), color, 2, cv2.LINE_4)
def drawGridlines(img):
	img[0:img.shape[0]:TILESIZE] = (0, 0, 0)
	img[:, 0:img.shape[1]:TILESIZE] = (0, 0, 0)
def _getImg(tile, targets, pos):
	tileName = {Tiles.FREE: 'Free',	Tiles.BULLDOZER: 'Bulldozer-up', Tiles.ROCK: 'Rock', Tiles.WALL: 'Wall'}[tile]
	if pos in targets:
		if tile == Tiles.ROCK:
			tileName = 'RockOnTarget'
		else: tileName = 'Target'
	return templates[tileName]
def drawDetectedLevel(tiles, targets):
	img = np.zeros((len(tiles) * TILESIZE, len(tiles[0]) * TILESIZE, 3), dtype='uint8')
	for y, col in enumerate(tiles):
		for x, tile in enumerate(col):
			template = _getImg(tile, targets, [x, y])
			blit(img, template, x * TILESIZE, y * TILESIZE)
	return img
# object detection --------------------------------------
def clipScreenshot(img):
	matched = cv2.matchTemplate(img, templates['Rock'], cv2.TM_CCOEFF_NORMED)
	posses = np.where(matched > 0.90)
	assert posses[0].shape[0] > 0
	minPos = min(posses[0]) % TILESIZE, min(posses[1]) % TILESIZE
	return img[minPos[0]:, minPos[1]:]
def getBestMatch(img) -> str:
	bestMatch = 'Free'
	bestVal = 0.0
	names = list(templates.keys())
	names.remove('Free')
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
	tiles = ['Free', 'Bulldozer-down', 'Bulldozer-left', 'Bulldozer-right', 'Bulldozer-up', 'Rock', 'RockOnTarget', 'Target', 'Wall', 'Wall2', 'Wall3', 'Wall4']
	assert set(templates.keys()) == set(tiles)
	matchName = getBestMatch(img)
	matched = Tiles.FREE
	if matchName in ['Bulldozer-down', 'Bulldozer-left', 'Bulldozer-right', 'Bulldozer-up']:
		matched = Tiles.BULLDOZER
	elif matchName in ['Rock', 'RockOnTarget']:
		matched = Tiles.ROCK
	elif matchName in ['Wall', 'Wall2', 'Wall3', 'Wall4']:
		matched = Tiles.WALL
	return matched, matchName in ['RockOnTarget', 'Target']
	
def detectLevel(img) -> tuple[list[list[Tiles]], list[list[int]]]:
	width, height = img.shape[1] // TILESIZE, img.shape[0] // TILESIZE
	tiles = [[Tiles.FREE for x in range(width)] for y in range(height)]
	targets = []
	for x in range(width):
		for y in range(height):
			tile, target = matchBlock(img[y * TILESIZE : y * TILESIZE + TILESIZE, x * TILESIZE : x * TILESIZE + TILESIZE])
			tiles[y][x] = tile
			if target:
				targets.append([x, y])
	return tiles, targets

# executing moves ------------------------------------
def executeMoves(moves: list[Moves], hwnd, duration=0.1):
	mapp = {Moves.UP: 'W', Moves.DOWN: 'S', Moves.RIGHT: 'D', Moves.LEFT: 'A'}
	for move in [ord(mapp[m]) for m in moves]:
		win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, move, 0)
		time.sleep(duration)
		win32gui.SendMessage(hwnd, win32con.WM_KEYUP, move, 0)

# run
def main():
	hwnd = findBulldozerWindow()
	img = getScreenshot(hwnd)

	img = clipScreenshot(img)
	tiles, targets = detectLevel(img)

	img = drawDetectedLevel(tiles, targets)
	cv2.imshow('BulldozeBot', img)

	moves = BotLogic.solveLevel(tiles, targets)
	executeMoves(moves, hwnd)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()