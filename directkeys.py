import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# Các mã phím
W = 0x11
S = 0x1F

# Hàm mô phỏng phím nhấn
def PressKey(key):
    extra = ctypes.c_ulong(0)
    ii_ = ctypes.c_ulong(0)
    # Thực hiện phím nhấn
    x = (ctypes.c_ulong * 2)()
    x[0] = ctypes.c_ulong(0)
    x[1] = ctypes.c_ulong(key)
    ctypes.windll.user32.SendInput(2, ctypes.byref(x), ctypes.sizeof(x))

# Hàm mô phỏng phím nhả
def ReleaseKey(key):
    extra = ctypes.c_ulong(0)
    ii_ = ctypes.c_ulong(0)
    # Thực hiện phím nhả
    x = (ctypes.c_ulong * 2)()
    x[0] = ctypes.c_ulong(0)
    x[1] = ctypes.c_ulong(key)
    ctypes.windll.user32.SendInput(2, ctypes.byref(x), ctypes.sizeof(x))

# Ví dụ sử dụng PressKey và ReleaseKey
if __name__ == '__main__':
    PressKey(W)
    time.sleep(1)
    ReleaseKey(W)
    time.sleep(1)
