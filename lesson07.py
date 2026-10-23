data = bytes([65, 66]) #表示把两个整数构造成一个两字节的字节串。每个整数必须在 0～255 范围内。

print(data)
print(list(data))
print(data.decode("utf-8"))