import win32gui, win32ui, win32con
import numpy as np
import cv2
import re

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
		dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
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

# run
hwnd = findBulldozerWindow()
print(f'Found window {win32gui.GetWindowText(hwnd)}')
LUX, LUY, RBX, RBY = win32gui.GetWindowRect(hwnd)

while True:
	img = getScreenshot(hwnd, RBX - LUX, RBY - LUY)
	cv2.imshow('BulldozeBot', img)
	cv2.waitKey(1)
	if cv2.getWindowProperty('BulldozeBot', cv2.WND_PROP_VISIBLE) < 1:        
		break

cv2.destroyAllWindows()
