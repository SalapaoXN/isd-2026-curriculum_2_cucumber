import sqlite3
import sqlite_vec
from contextlib import closing
from prepare_data import final

def Insert_db():
    vector_, max_text_, course_, meta_ = final()
    
    with closing(sqlite3.connect('new_course_database.db')) as conn:
        
        # โหลด Extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        # 2. Insert ตารางปกติ (course, meta_data, max_text)
        # ใช้ .to_sql() อ้างอิงผ่าน conn ได้เลย ไม่ต้องใช้ cursor
        course_.to_sql(name='course', con=conn, if_exists='fail', index=False)
        meta_.to_sql(name='meta_data', con=conn, if_exists='fail', index=False)
        max_text_.to_sql(name='max_text', con=conn, if_exists='fail', index=False)

        # 3. Insert ตาราง เสมือน (vector)
        with closing(conn.cursor()) as cursor:
            vector_data = []
            
            # วนลูปดึงข้อมูลจาก DataFrame ของเวกเตอร์
            for row in vector_.itertuples():
                # จัดเรียงให้ตรงกับคอลัมน์ (embed_byte, is_description, fk_Course)
                vector_data.append((row.embed_byte, row.is_description, row.fk_Course))
                
            cursor.executemany('''
                INSERT INTO vector (embed_byte, is_description, fk_Course) 
                VALUES (?, ?, ?)
            ''', vector_data)
            
        # บันทึกการเปลี่ยนแปลงทั้งหมด
        conn.commit()
        
    print("✅ Insert ข้อมูลลงฐานข้อมูลเสร็จสมบูรณ์!")

Insert_db()