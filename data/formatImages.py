import os
from PIL import Image, ImageFilter

"""formatImages.py processes the dataset and outputs a set of files into data/processed/ that are used by the other subsystems.
Most operations here are conceptually boring, and mostly deal with the file structure that the dataset uses.
Due to the impracticality of sending over gigabytes of photos, they are not included."""

ratio = 2   # [-] Aspect ratio of the images (w/h)
targetWidth = 64    # [px] Image width. Height is computed with the aspect ratio

# Horizontal Sobal operator for edge detection
edgeKernel = [1, 2, 1,
              0, 0, 0,
              -1, -2, -1]
kernel_filter = ImageFilter.Kernel((3, 3), edgeKernel, scale=1, offset=0)

bounds = []
with open("images_box.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        if len(e) == 5:
            bounds.append({'id': e[0], 'x1': int(e[1]), 'y1': int(e[2]), 'x2': int(e[3]), 'y2': int(e[4])})
    bounds = {x['id']: x for x in bounds}

families = []
with open("images_family_train.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        families.append({'id': e[0], 'family': " ".join(e[1:])})
with open("images_family_test.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        families.append({'id': e[0], 'family': " ".join(e[1:])})
with open("images_family_val.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        families.append({'id': e[0], 'family': " ".join(e[1:])})
families = {x['id']: x for x in families}

manufacturers = []
with open("images_manufacturer_train.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        manufacturers.append({'id': e[0], 'manufacturer': " ".join(e[1:])})
with open("images_manufacturer_test.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        manufacturers.append({'id': e[0], 'manufacturer': " ".join(e[1:])})
with open("images_manufacturer_val.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        manufacturers.append({'id': e[0], 'manufacturer': " ".join(e[1:])})
manufacturers = {x['id']: x for x in manufacturers}

variants = []
with open("images_variant_train.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        variants.append({'id': e[0], 'variant': " ".join(e[1:])})
with open("images_variant_test.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        variants.append({'id': e[0], 'variant': " ".join(e[1:])})
with open("images_variant_val.txt") as f:
    lines = f.read().split("\n")
    for line in lines:
        e = line.split(" ")
        variants.append({'id': e[0], 'variant': " ".join(e[1:])})
variants = {x['id']: x for x in variants}

images = os.listdir("images")
for filename in images:
    if len(filename) == 0:
        continue
    image = Image.open("images/" + filename)

    b = bounds[filename.split(".")[0]]

    width = b["x2"] - b["x1"]
    maxHeight = width / ratio
    height = b["y2"] - b["y1"]
    offset = (maxHeight - height) / 2
    image = image.crop((b["x1"], b["y1"] - offset, b["x2"], b["y2"] + offset))

    image = image.filter(kernel_filter)
    image = image.convert('L')  # Convert image to grayscale

    image = image.resize((targetWidth, int(targetWidth / ratio)))
    image.save("images/processed/" + filename.split(".")[
        0] + ".png")  # Optional for viewing the processed reference image. Highly space inefficient

    with open("images/processed/" + filename.split(".")[0] + ".txt", "w") as f:
        pixels = [str(image.getpixel((x, y))) for y in range(0, int(targetWidth / ratio)) for x in
                  range(0, targetWidth)]

        f.writelines(" ".join(pixels) + "\n")
        f.writelines(manufacturers[filename.split(".")[0]]["manufacturer"] + "\n")
        f.writelines(families[filename.split(".")[0]]["family"] + "\n")
        f.writelines(variants[filename.split(".")[0]]["variant"] + "\n")
    print(filename)
    break