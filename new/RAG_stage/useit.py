import sqlite3
import sqlite_vec
import numpy as np
from contextlib import closing
from model_to_vector import vector_model


class User:
    def __init__(self):
        self.model_vector = vector_model()
        
    def question(self, list_batch):
        '''prefer list of text'''
        self.vector = self.model_vector.word_to_vec(list_batch)
    def test_vector_search(self):
        with closing(sqlite3.connect('new_course_database.db')) as conn:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            with closing(conn.cursor()) as cursor:
                sql = '''
                    SELECT 
                        course.name_th,
                        vector.is_description, 
                        vector.distance
                    FROM vector
                    INNER JOIN course ON course.id=vector.fk_Course;
                    WHERE embed_byte MATCH ? AND k = 5

                '''
                for vec in self.vector:
                    vec_bytes = np.array(vec, dtype=np.float32).tobytes()
                    # ส่ง vec_bytes เข้าไปแทน ? โดยต้องมีลูกน้ำ (,) ต่อท้ายเสมอ
                    print(vec_bytes)
                    #cursor.execute(sql, (vec_bytes,))
                    #results = cursor.fetchall()
                    #print("ผลการค้นหา:")
                    #print(results)
                    #print("-" * 40)


obj = User()
obj.question(['วิชาที่เกี่ยวข้องกับคณิตศาสตร์' , 'วิชาการประมวลผลภาพสามารถเรียนตอนไหนได้บ้าง'])
obj.test_vector_search()