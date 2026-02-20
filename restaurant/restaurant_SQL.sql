PRAGMA foreign_keys = ON;

CREATE TABLE tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    seats INTEGER NOT NULL,
    zone TEXT
);

CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    phone TEXT,
    party_size INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL
);

CREATE TABLE reservation_tables (
    reservation_id INTEGER,
    table_id INTEGER,
    PRIMARY KEY (reservation_id, table_id),
    FOREIGN KEY (reservation_id) REFERENCES reservations(id) ON DELETE CASCADE,
    FOREIGN KEY (table_id) REFERENCES tables(id)
);

CREATE UNIQUE INDEX unique_reservation
ON reservations (customer_name, phone, start_time);
