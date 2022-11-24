import win32gui, win32ui, win32con
import numpy as np
import cv2
import re, os
from enum import IntEnum, auto

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
	if len(potentialWindowNames) >= 1:
		name = potentialWindowNames[0]
		return windowNames[name]
	else:
		raise RuntimeError('Could not find Bulldozer game window, windows found: ', windowNames.keys())

def getScreenshot(hwnd, W, H, cropped_x=8, cropped_y=30):
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

# structures --------------------------------------
class Tiles(IntEnum):
	WALL = auto()
	FREE = auto()
	ROCK = auto()
	BULLDOZER = auto()

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
	'Wall': cv2.imread(os.path.join('Templates', 'Wall.png'), cv2.IMREAD_COLOR),
	'Wall2': cv2.imread(os.path.join('Templates', 'Wall2.png'), cv2.IMREAD_COLOR),
}
# drawing -----------------------------------
def blit(img, src, x_offset: int, y_offset: int):
	img[y_offset:y_offset+src.shape[0], x_offset:x_offset+src.shape[1]] = src
def boxLocations(img, locs, templShape, color):
	for y, x in zip(*locs):
			cv2.rectangle(img, (x, y), (x + templShape[0], y + templShape[1]), color, 2, cv2.LINE_4)
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
			template = _getImg(tile, targets, (x, y))
			blit(img, template, x * TILESIZE, y * TILESIZE)
	return img
# object detection --------------------------------------
def getBestMatch(img) -> str:
	bestMatch = list(templates.keys())[0]
	bestVal = 0.0
	for name in templates.keys():
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
	tiles = ['Free', 'Bulldozer-down', 'Bulldozer-left', 'Bulldozer-right', 'Bulldozer-up', 'Rock', 'RockOnTarget', 'Target', 'Wall', 'Wall2']
	assert set(templates.keys()) == set(tiles)
	matchName = getBestMatch(img)
	matched = Tiles.FREE
	if matchName in ['Bulldozer-down', 'Bulldozer-left', 'Bulldozer-right', 'Bulldozer-up']:
		matched = Tiles.BULLDOZER
	elif matchName in ['Rock', 'RockOnTarget']:
		matched = Tiles.ROCK
	elif matchName in ['Wall', 'Wall2']:
		matched = Tiles.WALL
	return matched, matchName in ['RockOnTarget', 'Target']
	
def detectLevel(img) -> tuple[list[list[Tiles]], list[tuple[int]]]:
	width, height = img.shape[1] // TILESIZE, img.shape[0] // TILESIZE
	tiles = [[Tiles.FREE for x in range(width)] for y in range(height)]
	targets = []
	for x in range(width):
		for y in range(height):
			tile, target = matchBlock(img[y * TILESIZE : y * TILESIZE + TILESIZE, x * TILESIZE : x * TILESIZE + TILESIZE])
			tiles[y][x] = tile
			if target:
				targets.append((x, y))
	return tiles, targets

# run
hwnd = findBulldozerWindow()
print(f'Found window {win32gui.GetWindowText(hwnd)}')
LUX, LUY, RBX, RBY = win32gui.GetWindowRect(hwnd)

while True:
	img = getScreenshot(hwnd, RBX - LUX, RBY - LUY)
	tiles, targets = detectLevel(img)
	img = drawDetectedLevel(tiles, targets)
	print('WALL', sum([t.count(Tiles.WALL) for t in tiles]))
	print('FREE', sum([t.count(Tiles.FREE) for t in tiles]))
	print('ROCK', sum([t.count(Tiles.ROCK) for t in tiles]))
	print('BULLDOZER', sum([t.count(Tiles.BULLDOZER) for t in tiles]))
	print(targets)
	
	cv2.imshow('BulldozeBot', img)
	if cv2.waitKey(20) == ord('q') or cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:
		break

cv2.destroyAllWindows()
