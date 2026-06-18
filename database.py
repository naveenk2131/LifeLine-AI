"""
database.py — LifeLine AI
Manages the local SQLite database for blood donors.
Creates `donors.db` with mock data and provides query functions.
"""

import sqlite3
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(__file__), "donors.db")


# ─────────────────────────────────────────────────────────────
# Schema & Seed Data
# ─────────────────────────────────────────────────────────────

def create_database() -> None:
    """
    Creates the donors table (if it doesn't exist) and seeds
    5 mock donors covering different blood groups and cities.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS donors (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT    NOT NULL,
            phone             TEXT    NOT NULL UNIQUE,
            blood_group       TEXT    NOT NULL,
            city              TEXT    NOT NULL,
            last_donation_date TEXT   NOT NULL
        )
    """)

    # Mock donor seed data
    mock_donors = [
        ("Arjun Sharma",   "+919876543210", "O-",  "Salem",   "2023-11-15"),
        ("Priya Nair",     "+919876543211", "O-",  "Salem",   "2024-01-20"),
        ("Karthik Raj",    "+919876543212", "A+",  "Chennai", "2024-03-05"),
        ("Divya Menon",    "+919876543213", "B+",  "Salem",   "2023-08-22"),
        ("Ravi Subramaniam","+919876543214","O-",  "Coimbatore","2024-02-10"),
    ]

    # Insert only if the table is empty (avoid duplicate seeding)
    cursor.execute("SELECT COUNT(*) FROM donors")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.executemany(
            "INSERT INTO donors (name, phone, blood_group, city, last_donation_date) "
            "VALUES (?, ?, ?, ?, ?)",
            mock_donors
        )
        print(f"[DB] Seeded {len(mock_donors)} mock donors.")
    else:
        print(f"[DB] Database already contains {count} donors — skipping seed.")

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Query Functions
# ─────────────────────────────────────────────────────────────

def get_donors_by_blood_group_and_city(blood_group: str, city: str) -> list[dict]:
    """
    Returns a list of donors matching the given blood group AND city.
    Both comparisons are case-insensitive.

    Args:
        blood_group: e.g. "O-", "A+", "B+"
        city:        e.g. "Salem", "Chennai"

    Returns:
        List of donor dicts with keys: id, name, phone, blood_group, city,
        last_donation_date.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # enables dict-like access
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, phone, blood_group, city, last_donation_date
        FROM   donors
        WHERE  LOWER(blood_group) = LOWER(?)
          AND  LOWER(city)        = LOWER(?)
        """,
        (blood_group, city)
    )
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_all_donors() -> list[dict]:
    """Returns every donor in the database (for the admin dashboard view)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, phone, blood_group, city, last_donation_date FROM donors"
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ─────────────────────────────────────────────────────────────
# Entry-point: run this file directly to initialise the DB
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_database()
    donors = get_all_donors()
    print(f"\nAll donors in DB ({len(donors)} total):")
    for d in donors:
        print(
            f"  [{d['id']}] {d['name']} | {d['blood_group']} | "
            f"{d['city']} | Last donated: {d['last_donation_date']}"
        )
