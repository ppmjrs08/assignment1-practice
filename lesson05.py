text = "中"

print("UTF-8:", len(text.encode("utf-8")))
print("UTF-16:", len(text.encode("utf-16-le")))
print("UTF-32:", len(text.encode("utf-32-le")))