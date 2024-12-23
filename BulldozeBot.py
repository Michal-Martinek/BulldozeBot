import win32gui, win32ui, win32con
import numpy as np
import cv2
import re, os

import subprocess
import threading
import queue
import signal

os.chdir(os.path.dirname(__file__))

import BotLogic
from BotLogic import State
from Classes import Pos, Tiles, Board, TILES_TYPE, Moves, Comms

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
def blit(img, src, y_offset: int, x_offset: int):
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
	img[::TILESIZE] = (0, 0, 0)
	img[:, ::TILESIZE] = (0, 0, 0)

# object detection --------------------------------------
def clipScreenshot(img) -> tuple[np.ndarray, tuple[int, int]]:
	matched = cv2.matchTemplate(img, templates['Rock'], cv2.TM_CCOEFF_NORMED)
	posses = np.array( np.where(matched > 0.90) )
	assert posses.size, 'Could not find any rocks, the level is probably solved'
	minPos = np.min(posses, axis=1) % TILESIZE
	img = img[minPos[0]:, minPos[1]:]
	height, width = img.shape[0] // TILESIZE, img.shape[1] // TILESIZE
	return img[:height * TILESIZE, :width * TILESIZE], (height, width)
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
	
def detectLevel(img: np.ndarray, height, width) -> tuple[Board, list[Pos]]:
	blocks = img.reshape((height, TILESIZE, width, TILESIZE, 3))
	tiles: Board = np.zeros((height, width), dtype=TILES_TYPE)
	targets = []
	for y in range(height):
		for x in range(width):
			tile, target = matchBlock(blocks[y, :, x])
			tiles[y, x] = tile
			if target:
				targets.append(Pos(y, x))
	return tiles, targets

class GUI:
	def __init__(self, windowName):
		self.windowName = windowName
		self.repeat = True
	def init(self):
		self.hwnd: int = -1
		self.repeat = False
		self.comms = Comms()
		self.solveThread: threading.Thread = None
	def stop(self):
		self.comms.stopEvent.set()
		return False
	def joinThread(self):
		assert self.comms.stopEvent.is_set()
		self.solveThread.join(1)
		if self.solveThread.is_alive():
			print('ERROR: closing SolveThread timed out, exiting')
			os.kill(os.getpid(), signal.SIGTERM)
	def close(self):
		assert self.comms.stopEvent.is_set()
		try:
			cv2.destroyWindow(self.windowName)
		except cv2.error:
			pass
		self.joinThread()

	# drawing ----------------------------------------
	@staticmethod
	def getTemplateName(tile: Tiles, isTarget: bool, forbidden: bool):
		tileName = {Tiles.FREE: ['Free', 'Free-forbidden'][forbidden],	Tiles.BULLDOZER: ['Bulldozer-up', 'Bulldozer-forbidden'][forbidden], Tiles.ROCK: 'Rock', Tiles.WALL: 'Wall'}[tile]
		if isTarget:
			if tile == Tiles.BULLDOZER:
				tileName = 'Bulldozer-target'
			elif tile == Tiles.ROCK:
				tileName = 'RockOnTarget'
			else: tileName = 'Target'
		return tileName
	def getTemplateImg(self, pos: Pos, state):
		tile = state.tiles[pos]
		if pos == state.bulldozerPos:
			tile = Tiles.BULLDOZER
		if pos in state.rocks:
			tile = Tiles.ROCK
		name = self.getTemplateName(tile, pos in state.targets, bool(state.forbidden[pos]))
		return templates[name]
	def drawTiles(self, state: State):
		img = np.zeros((len(state.tiles) * TILESIZE, len(state.tiles[0]) * TILESIZE, 3), dtype='uint8')
		for pos in Pos.iterBoard(state.tiles, inner=False):
			template = self.getTemplateImg(pos, state)
			blit(img, template, *pos * TILESIZE)
		return img
	def drawState(self, state):
		return self.drawTiles(state)

	# ---------------------------------------------
	def getNewestState(self):
		state = None
		while True:
			try:
				state = self.comms.stateQueue.get_nowait()
			except queue.Empty:
				return state
	
	def advanceLevel():
		raise NotImplementedError
	
	def checkDisplay(self, wait_ms=20) -> bool:
		key = cv2.waitKey(wait_ms)
		if cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) < 1:
			return self.stop()
		if key in map(ord, 'qr\x1b'):
			if key == ord('r'):
				self.repeat = True
			return self.stop()
		return True
	def redrawDisplay(self):
		if (state := self.getNewestState()) is None: return
		img = self.drawState(state)
		cv2.imshow(self.windowName, img)
	def checkSolveThread(self):
		if not self.solveThread.is_alive():
			return self.stop()
		if self.comms.doneEvent.is_set():
			self.comms.stopEvent.set()
			self.joinThread()
	
	# loops ---------------------------------------------
	def executeMovesLoop(self, moves: list[Moves], dontFinish=False):
		assert not self.comms.stopEvent.is_set()
		mapp = {Moves.UP: 'W', Moves.DOWN: 'S', Moves.RIGHT: 'D', Moves.LEFT: 'A'}
		if dontFinish: moves = moves[:-1]
		for move in [ord(mapp[m]) for m in moves]:
			win32gui.PostMessage(self.hwnd, win32con.WM_KEYDOWN, move, 0)
			if not self.checkDisplay():
				return
			win32gui.SendMessage(self.hwnd, win32con.WM_KEYUP, move, 0)
	def waitInLoop(self, solving: bool):
		while not self.comms.stopEvent.is_set():
			if (solving):
				self.redrawDisplay()
				self.checkSolveThread()
			self.checkDisplay()
	def solveAndExecute(self, startState):
		self.solveThread = threading.Thread(target=BotLogic.solveLevel, args=(startState, self.comms), name='SolveThread')
		self.solveThread.start()
		self.waitInLoop(solving=True)
		if self.comms.doneEvent.is_set():
			moves = self.comms.movesQueue.get_nowait()
			self.comms.stopEvent.clear()
			self.redrawDisplay()
			print(f'INFO: found a solution with {len(moves)} moves')
			self.executeMovesLoop(moves)
			self.waitInLoop(solving=False)
		
	def readGameInput(self) -> State:
		self.hwnd = findBulldozerWindow()
		img = getScreenshot(self.hwnd)
		
		img, dims = clipScreenshot(img)
		tiles, targets = detectLevel(img, *dims)
		state = BotLogic.prepareLevel(tiles, targets)
		return state
	def mainLoop(self):
		while self.repeat:
			self.init()
			state = self.readGameInput()
			self.solveAndExecute(state)
		self.close()

def main():
	gui = GUI('Bulldozer Bot')
	gui.mainLoop()
	
if __name__ == '__main__':
	main()
