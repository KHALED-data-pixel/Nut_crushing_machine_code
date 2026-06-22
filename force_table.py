FORCE_LOOKUP = {
    "argan":  [(0,  13, 400), (13, 16, 500), (16, 999, 600)],
    "almond": [(0,  18, 250), (18, 23, 300), (23, 999, 380)],
    "peanut": [(0,  10,  80), (10, 13, 100), (13, 999, 140)],
}

DEFAULT_FORCE = 500.0


def get_force(nut_class: str, width_mm: float) -> float:
    table = FORCE_LOOKUP.get(nut_class.lower())
    if not table:
        return DEFAULT_FORCE
    for low, high, force in table:
        if low <= width_mm < high:
            return float(force)
    return DEFAULT_FORCE


if __name__ == "__main__":
    print(f"{'Nut':<10} {'Min mm':<10} {'Max mm':<10} Force N")
    print("-" * 40)
    for nut, bands in FORCE_LOOKUP.items():
        for lo, hi, f in bands:
            print(f"{nut:<10} {lo:<10} {'inf' if hi==999 else hi:<10} {f}")

    print("\nSelf-test:")
    for cls, w in [("argan",12),("argan",14),("almond",20),("peanut",11),("unknown",10)]:
        print(f"  {cls:<10} {w:5.1f} mm -> {get_force(cls,w):.0f} N")
