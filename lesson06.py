text = "A"
encoded = text.encode("utf-8")

print(list(encoded))
print(encoded.decode("utf-8"))

first_byte = encoded[:1] #表示从开头取到位置 1 之前，也就是只取第一个字节，得到的仍然是字节串。
print(list(first_byte))
print(first_byte.decode("utf-8")) # A的首字节可以独立解码；中的首字节不能。