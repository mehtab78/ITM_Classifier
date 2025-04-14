import os

image_dir = "visual7w-images"
with open("v7w.TrainImages.itm.txt") as f:
    lines = f.readlines()

missing = []
for line in lines:
    img = line.strip().split("\t")[0]
    if not os.path.exists(os.path.join(image_dir, img)):
        missing.append(img)

print(f"{len(missing)} missing images:")
for m in missing[:10]:
    print(m)
