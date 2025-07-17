import psycopg
from psycopg import sql

# Connect to the database
conn = psycopg.connect("postgresql://Admin:12345678@localhost:5432/mydatabase")
cur = conn.cursor()

# Drop the table if it exists
cur.execute("""
    DROP TABLE IF EXISTS langgraph_events
""")
print("Dropped existing table 'langgraph_events'.")

# Recreate the table with a 'timestamp' column instead of 'created_at'
cur.execute("""
    CREATE TABLE langgraph_events (
        id SERIAL PRIMARY KEY,
        thread_id UUID NOT NULL,
        event JSONB NOT NULL,
        timestamp TIMESTAMP DEFAULT NOW()
    )
""")
print("Created new table 'langgraph_events' with 'timestamp' column.")

# Commit changes and close
conn.commit()
cur.close()
conn.close()
