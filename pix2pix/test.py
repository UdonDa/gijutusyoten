import glob

a = glob.glob("../datasets/non_renge_short_paste_mask/*")
b = glob.glob("../datasets/non_renge_short/*")
print(a[3])
print(len(b))
