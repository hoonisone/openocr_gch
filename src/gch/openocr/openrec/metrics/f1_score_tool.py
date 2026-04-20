import re


def parse_char_count_line(line):
    s = line.strip()
    if not s:
        return None

    if "\t" in s:
        a, b = s.split("\t", 1)
        a, b = a.strip(), b.strip()
        if re.fullmatch(r"\d+", b):
            return a, int(b)
        if re.fullmatch(r"\d+", a):
            return b, int(a)

    parts = [p for p in re.split(r"[: ,]+", s) if p]
    if len(parts) >= 2:
        if re.fullmatch(r"\d+", parts[1]):
            return parts[0], int(parts[1])
        if re.fullmatch(r"\d+", parts[0]):
            return parts[1], int(parts[0])

    return None


def load_char_train_count(path):
    char_count = {}
    if not path:
        return char_count

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_char_count_line(line)
                if parsed is None:
                    continue
                ch, cnt = parsed
                char_count[ch] = cnt
    except OSError:
        return {}
    return char_count


def make_bins_from_edges(edges):
    if edges is None or len(edges) < 2:
        return []
    edges = sorted(set(int(x) for x in edges))
    bins = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        bins.append((f"{low}~{high}", low, high))
    bins.append((f"{edges[-1]}+", edges[-1], None))
    return bins


def in_bin(count, low, high):
    if high is None:
        return count >= low
    return (count >= low) and (count < high)


def build_char_bin_summary(
    per_char_confusion,
    total_char_events,
    char_train_count,
    char_bin_edges,
    eps=1e-5,
):
    bins = make_bins_from_edges(char_bin_edges)
    if not char_train_count or not bins:
        return {}

    bin_report = {}
    for label, low, high in bins:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        num_chars = 0

        for ch, cnt in char_train_count.items():
            if not in_bin(cnt, low, high):
                continue
            num_chars += 1
            counts = per_char_confusion[ch]
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            tn = total_char_events - tp - fp - fn

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        f1 = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
        bin_report[label] = {
            "num_chars": int(num_chars),
            "tp": int(total_tp),
            "tn": int(total_tn),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "f1": f1,
        }

    return {"per_char_bin_confusion": bin_report}
