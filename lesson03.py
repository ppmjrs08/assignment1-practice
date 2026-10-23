character = chr(0) #使用chr函数将码点转换为字符并保存到character变量中
print(character)
print(len(character))
print(repr(character))

text = "A" + character + "B"

print(text)
print(repr(text))
print(len(text))
