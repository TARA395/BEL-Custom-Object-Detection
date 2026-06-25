"""
verify_labels.py

Cross-checks every YOLO-format label .txt file against classes.txt
and prints out, for every image, which class name(s) its bounding
box(es) currently point to.

Use this to quickly scan for anything that looks wrong (e.g. a file
you remember labeling "Penguin" showing up as "Kangaroo") without
opening every image in LabelImg.

USAGE:
    Just edit the two paths below (LABELS_DIR, IMAGES_DIR) if needed,
    then run:

        python verify_labels.py

    It will also write a file called "label_report.txt" next to this
    script with the same output, so you can open it in Notepad and
    scroll through it more easily.
"""

import os

# ---- EDIT THESE IF YOUR PATHS ARE DIFFERENT ----
LABELS_DIR = r"D:\BEL\my_dataset\animal_labels\labels_train"
IMAGES_DIR = r"D:\BEL\my_dataset\animal_images\images_train"
CLASSES_FILE = os.path.join(LABELS_DIR, "classes.txt")
REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_report.txt")
# -------------------------------------------------

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def load_classes(classes_file):
    if not os.path.isfile(classes_file):
        raise FileNotFoundError(f"Could not find classes.txt at: {classes_file}")
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip() != ""]
    return classes


def find_matching_image(label_filename, images_dir):
    """Given 'image_001.txt', try to find 'image_001.jpg' / .png / etc."""
    base_name = os.path.splitext(label_filename)[0]
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.isfile(candidate):
            return base_name + ext
    return None  # no matching image found


def parse_label_file(label_path, classes):
    """
    Returns a list of (class_name_or_error, raw_line) tuples for a YOLO
    label file. Flags out-of-range or malformed indices instead of
    crashing, since that's exactly the kind of corruption we're hunting
    for.
    """
    results = []
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    if not lines:
        results.append(("<<EMPTY FILE - no boxes>>", ""))
        return results

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        class_idx_str = parts[0]
        try:
            class_idx = int(class_idx_str)
        except ValueError:
            results.append((f"<<MALFORMED LINE - class index not an integer: '{class_idx_str}'>>", line))
            continue

        if class_idx < 0 or class_idx >= len(classes):
            results.append((f"<<OUT OF RANGE INDEX {class_idx} - classes.txt only has {len(classes)} entries>>", line))
        else:
            results.append((classes[class_idx], line))

    return results


def main():
    print(f"Loading classes from: {CLASSES_FILE}")
    classes = load_classes(CLASSES_FILE)
    print("Classes (index -> name):")
    for i, c in enumerate(classes):
        print(f"  {i} -> {c}")
    print()

    if not os.path.isdir(LABELS_DIR):
        raise FileNotFoundError(f"Labels directory not found: {LABELS_DIR}")

    label_files = sorted(f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt") and f.lower() != "classes.txt")

    if not label_files:
        print("No label .txt files found. Check LABELS_DIR path.")
        return

    report_lines = []
    flagged_lines = []  # anything suspicious goes here too, for a quick-look summary

    for label_file in label_files:
        label_path = os.path.join(LABELS_DIR, label_file)
        matched_image = find_matching_image(label_file, IMAGES_DIR)
        image_display = matched_image if matched_image else "<<NO MATCHING IMAGE FOUND>>"

        parsed = parse_label_file(label_path, classes)
        class_names = [p[0] for p in parsed]
        line_str = f"{label_file:30s}  image: {image_display:35s}  ->  {', '.join(class_names)}"
        report_lines.append(line_str)

        # Flag anything with an issue marker so it's easy to find later
        if matched_image is None or any(name.startswith("<<") for name in class_names):
            flagged_lines.append(line_str)

    full_report = []
    full_report.append("=== FULL LABEL REPORT ===")
    full_report.extend(report_lines)
    full_report.append("")
    full_report.append("=== FLAGGED / SUSPICIOUS ENTRIES (review these first) ===")
    if flagged_lines:
        full_report.extend(flagged_lines)
    else:
        full_report.append("None found - no missing images or malformed/out-of-range indices detected.")

    output_text = "\n".join(full_report)
    print(output_text)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\nFull report also saved to: {REPORT_FILE}")
    print(f"Total label files checked: {len(label_files)}")
    print(f"Flagged entries: {len(flagged_lines)}")


if __name__ == "__main__":
    main()
