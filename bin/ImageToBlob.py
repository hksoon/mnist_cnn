import script as sc
import struct
import array

TableNm = "mnist"

rows = sc.selectData(TableNm)

def bcdDigits(chars):
    for char in chars:
        char = ord(char)
        return char
i=0

print(rows)

for row in rows:
    a = struct.unpack('8B', row[380])
    print(a)

