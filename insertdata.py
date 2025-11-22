import psycopg2
import os

# ---------------------------------------
# CONNECT TO DATABASE
# ---------------------------------------
conn = psycopg2.connect(
    "postgresql://neondb_owner:npg_dL4STO9vmgnA@ep-jolly-shape-abv7053m-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)
cur = conn.cursor()

# ---------------------------------------
# ALTER TABLE TO ADD image_path COLUMN
# ---------------------------------------
alter_table_query = """
ALTER TABLE products
ADD COLUMN IF NOT EXISTS image_path TEXT;
"""
cur.execute(alter_table_query)
conn.commit()

# ---------------------------------------
# PRODUCTS TO INSERT
# ---------------------------------------
products = [
    {
        "name": "Product1 - Hydrating Face Cream",
        "skin_hair_type": "Dry, Normal",
        "key_ingredients": "Hyaluronic Acid, Glycerin, Shea Butter",
        "benefits": "Deep hydration, restores skin barrier, smooths fine lines",
        "usage": "Apply morning and night to cleansed face and neck. Use after serum.",
        "precautions": "For external use only. Patch test for sensitive skin.",
        "price": "180 MAD",
        "size": "50 ml",
        "image_path": "images/p1.jpg"
    },
    {
        "name": "Product2 - Mattifying Gel Cream",
        "skin_hair_type": "Oily, Combination",
        "key_ingredients": "Niacinamide, Zinc PCA, Salicylic Acid (0.5%)",
        "benefits": "Controls shine, minimizes pores, reduces acne breakouts",
        "usage": "Apply a pea-sized amount in the morning after toner. Reapply as needed.",
        "precautions": "Avoid contact with eyes. If irritation occurs, discontinue use.",
        "price": "160 MAD",
        "size": "40 ml",
        "image_path": "images/p2.jpg"
    },
    {
        "name": "Product3 - Soothing Night Serum",
        "skin_hair_type": "Sensitive, Redness-prone",
        "key_ingredients": "Centella Asiatica (Cica), Panthenol, Allantoin",
        "benefits": "Calms irritation, supports skin recovery, improves texture overnight",
        "usage": "Apply 2-3 drops on cleaned skin before moisturizer at night.",
        "precautions": "Perform patch test if highly reactive.",
        "price": "220 MAD",
        "size": "30 ml",
        "image_path": "images/p3.jpg"
    },
    {
        "name": "Product4 - Brightening Vitamin C Serum",
        "skin_hair_type": "Dull, Pigmented, All skin types",
        "key_ingredients": "Vitamin C (L-ascorbic acid 10%), Ferulic Acid, Vitamin E",
        "benefits": "Reduces dark spots, brightens complexion, antioxidant protection",
        "usage": "Apply in the morning before sunscreen. Store away from direct sunlight.",
        "precautions": "May increase sun sensitivity; use SPF daily.",
        "price": "240 MAD",
        "size": "30 ml",
        "image_path": "images/p4.jpg"
    },
    # Add other products similarly, with image_path
]

# ---------------------------------------
# INSERT PRODUCTS (WITH IMAGE PATH)
# ---------------------------------------
insert_query = """
INSERT INTO products 
(name, skin_hair_type, key_ingredients, benefits, usage, precautions, price, size, image_path)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

for p in products:
    cur.execute(insert_query, (
        p["name"],
        p["skin_hair_type"],
        p["key_ingredients"],
        p["benefits"],
        p["usage"],
        p["precautions"],
        p["price"],
        p["size"],
        p["image_path"]
    ))

conn.commit()
print("✅ All products inserted successfully with images!")

cur.close()
conn.close()
