# Các giá trị đã cho
a = 1
b = 2
c = 0
d = -2
e = -4
f = 5

# Tìm giá trị nhỏ nhất và lớn nhất
min_value = min(a, b, c, d, e, f)
max_value = max(a, b, c, d, e, f)

min_value -= 1
max_value += 1

range_value = max_value - min_value

# Tính phần trăm cho từng giá trị
a_percent = (a - min_value) / range_value * 100
b_percent = (b - min_value) / range_value * 100
c_percent = (c - min_value) / range_value * 100
d_percent = (d - min_value) / range_value * 100
e_percent = (e - min_value) / range_value * 100
f_percent = (f - min_value) / range_value * 100

# In kết quả
print(f"a: {a_percent}%")
print(f"b: {b_percent}%")
print(f"c: {c_percent}%")
print(f"d: {d_percent}%")
print(f"e: {e_percent}%")
print(f"f: {f_percent}%")