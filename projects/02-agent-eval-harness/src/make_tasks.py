"""Generate the evaluation set: messy commerce listings with known ground truth.

    python -m src.make_tasks --n 150 --out data/tasks.jsonl

Synthetic, for the same reason as project 01: the ground truth is known exactly, so a
scoring bug is detectable rather than being absorbed into a plausible-looking number.
Listings are deliberately noisy in the ways real marketplace copy is — inconsistent
casing, SEO keyword stuffing, seller boilerplate, emoji, and fields that are genuinely
absent rather than merely hard to find.

The dev/heldout split is written here and enforced in the harness. Prompt iteration
touches `dev` only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BRANDS = ["Aerolite", "Northvale", "Kestrel", "BluePine", "Harrow & Co", "Vantage",
          "Selkie", "Ironwood", "Marlow", "Thornbury"]
PRODUCTS = ["running shoe", "rain jacket", "duffel bag", "wool sweater", "water bottle",
            "hiking boot", "fleece vest", "trail pack", "sun hoodie", "camp chair"]
COLORS = ["charcoal", "slate blue", "moss green", "burnt orange", "off white",
          "navy", "oxblood", "sand", "forest", "graphite"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL", "8", "9.5", "10", "11.5"]
MATERIALS = ["merino wool", "recycled polyester", "ripstop nylon", "organic cotton",
             "gore-tex", "cordura", "bamboo viscose"]

NOISE_PREFIX = [
    "🔥 FLASH SALE 🔥 ", "", "NEW ARRIVAL | ", "⭐⭐⭐⭐⭐ BEST SELLER ⭐⭐⭐⭐⭐ ", "",
    "[SHIPS IN 24H] ", "", "*** CLEARANCE *** ",
]
NOISE_SUFFIX = [
    "\n\nFree returns within 30 days. Ships from our warehouse.",
    "\n\nKeywords: outdoor gear hiking camping travel lightweight durable premium quality",
    "",
    "\n\nSee our other listings for matching items!!!",
    "\n\nNote: colors may vary slightly due to monitor settings.",
    "",
    "\n\n#outdoors #gear #adventure #sustainable #madetolast",
]
TEMPLATES = [
    "{prefix}{brand} {product} - {color} - Size {size} | {material} | ${price}{suffix}",
    "{prefix}{brand.upper} {product.title} ({color}), sz {size}. Made from {material}. "
    "Only ${price}!{suffix}",
    "{prefix}Brand new {color} {product} by {brand}. {material} construction. "
    "Size: {size}. Price: USD {price}{suffix}",
    "{prefix}{product} // {brand} // {color} // {size} // {material} // {price} USD{suffix}",
]


def _fill(template: str, **kw) -> str:
    out = template
    for key, value in kw.items():
        out = out.replace("{" + key + ".upper}", str(value).upper())
        out = out.replace("{" + key + ".title}", str(value).title())
        out = out.replace("{" + key + "}", str(value))
    return out


def make_item(i: int, rng: random.Random) -> dict:
    brand, product = rng.choice(BRANDS), rng.choice(PRODUCTS)
    color, size, material = rng.choice(COLORS), rng.choice(SIZES), rng.choice(MATERIALS)
    price = f"{rng.randint(19, 340)}.{rng.choice(['00', '95', '99', '50'])}"

    truth = {"brand": brand, "product_type": product, "color": color,
             "size": size, "material": material, "price_usd": price}

    # A fifth of items are missing a field entirely. Extraction systems that never see
    # an absent field learn to always produce something, which is the worst failure mode
    # because it is invisible in aggregate accuracy.
    if rng.random() < 0.2:
        dropped = rng.choice(["material", "color", "size"])
        truth[dropped] = None

    listing = _fill(
        rng.choice(TEMPLATES),
        prefix=rng.choice(NOISE_PREFIX), suffix=rng.choice(NOISE_SUFFIX),
        brand=brand, product=product, price=price,
        color=truth["color"] or "", size=truth["size"] or "",
        material=truth["material"] or "",
    )
    listing = listing.replace(" -  - ", " - ").replace("()", "").replace("//  //", "//")

    return {
        "id": f"item-{i:04d}",
        "split": "dev" if i % 5 == 0 else "heldout",
        "listing": listing,
        "fields": list(truth.keys()),
        "truth": truth,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("data/tasks.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = [make_item(i, rng) for i in range(args.n)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(it) for it in items) + "\n")

    dev = sum(1 for i in items if i["split"] == "dev")
    print(f"wrote {args.out}: {len(items)} items ({dev} dev, {len(items) - dev} heldout)")


if __name__ == "__main__":
    main()
