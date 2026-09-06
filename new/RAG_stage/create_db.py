'''Create Your Schema'''
import sqlite3
import sqlite_vec
from contextlib import closing

def create_schema():
    with closing(sqlite3.connect('new_course_database.db')) as conn:
        
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        with closing(conn.cursor()) as cursor:
            
            # 1. ตาราง Course 
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS course (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    name_th TEXT,
                    name_en TEXT,
                    credits TEXT,
                    year INTEGER,
                    semester INTEGER,
                    category TEXT,
                    type TEXT,
                    prerequisite TEXT,
                    flexible_year_semester TEXT,
                    note TEXT,
                    desc_th TEXT , 
                    desc_en TEXT
                )
            ''')

            # 2. ตาราง meta data (M:1 ไปหา Course)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS meta_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program TEXT,
                    source_filename TEXT,
                    source_page INTEGER,
                    document_category TEXT,
                    fk_Course INTEGER,
                    FOREIGN KEY (fk_Course) REFERENCES course(id)
                )
            ''')

            # 3. ตาราง vector (M<=2 ไปหา Course)
            # ใช้ embedding เป็นตัวแทนของ vector_course และ vector_description
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS vector USING vec0(
                    embed_byte float[384],
                    +is_description INTEGER,
                    +fk_Course INTEGER
                )
            ''')

            # 4. ตาราง max_text (1:1 ไปหา vector)
            # fk_vector จะอ้างอิงไปยัง rowid ของตาราง vector (Virtual Table)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS max_text (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    is_description INTEGER,
                    fk_vector INTEGER,
                    fk_Course INTEGER,
                    FOREIGN KEY (fk_Course) REFERENCES course(id)
                )
            ''')

        conn.commit()
        
    print(" สร้างตารางตาม Schema สำเร็จแล้ว!")

if __name__ == "__main__":
    create_schema()