import pickle, torch

def max_abs_merge(delta_a: dict, delta_b: dict) -> dict:
    """
    Combine two LoRA-delta dicts by keeping, for each element, the value
    with the larger absolute magnitude.  Sign is preserved.
    """
    merged = {}

    keys = set(delta_a) | set(delta_b)
    for name in keys:
        if name in delta_a and name in delta_b:
            # ensure same dtype / device
            t1, t2 = delta_a[name], delta_b[name]
            if t1.shape != t2.shape:
                raise ValueError(f"Shape mismatch for {name}: {t1.shape} vs {t2.shape}")

            # |t1| >= |t2| → take t1, else take t2
            mask = (t1.abs() >= t2.abs())
            merged[name] = torch.where(mask, t1, t2)

        elif name in delta_a:
            merged[name] = delta_a[name].clone()
        else:
            merged[name] = delta_b[name].clone()

    return merged


# ---------- usage ----------
with open("../MEDICINE/med_lora_weights.pkl", "rb") as f:
    med = pickle.load(f)
with open("../MATH/math_lora_weights.pkl", "rb") as f:
    math = pickle.load(f)

merged = max_abs_merge(med, math)

with open("combined_maxabs.pkl", "wb") as f:
    pickle.dump(merged, f)

print("Wrote combined_maxabs.pkl ✓")

