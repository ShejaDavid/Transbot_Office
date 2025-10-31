import os

labels_dir = "train/labels"
max_boxes_per_image = 2  # keep at most 2 boxes per image
target_class = 4  # Drycell

for fname in os.listdir(labels_dir):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(labels_dir, fname)
    with open(path) as f:
        lines = f.readlines()

    # Keep only up to N bounding boxes for the target class
    filtered = []
    count = 0
    for line in lines:
        cls = int(line.split()[0])
        if cls == target_class:
            if count < max_boxes_per_image:
                filtered.append(line)
            count += 1
        else:
            filtered.append(line)

    # Write back modified labels
    with open(path, "w") as f:
        f.writelines(filtered)

print(f"✅ Limited class {target_class} to {max_boxes_per_image} boxes per image.")
