'''เตรียมข้อมูลเพื่อ insert เข้า database'''

import os
import json
from json_to_text import json_to_text , json_to_desc
import pandas as pd
from model_to_vector import vector_model
import numpy as np
# your Json folder
your_path = r'C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\consolidated_outputs\consolidated_outputs'



def max_text(path=your_path):
    '''
    Purpose Data for max text table
    Call function json_to_text() from json_to_text.py
    Input as folder of json prefer suffix 'full'
    Output as Pandas DataFrame
    '''
    file_path = path
    file_name = os.listdir(file_path)
    
    # 1. สร้าง List เปล่าเพื่อเก็บข้อมูลเป็นชุดๆ
    data_list = []
    id_ = 1
    for file in file_name: # เปลี่ยนชื่อตัวแปรเป็น file เพื่อไม่ให้ซ้ำ
        if 'full' not in os.path.basename(file):
            continue
            
        with open(os.path.join(file_path, file), encoding='utf-8', mode='r') as f:
            json_data = json.loads(f.read())
            
            # ใช้ .get() เผื่อกรณีไฟล์ JSON ไม่มีคีย์ 'courses'
            for course in json_data.get('courses', []): 
                
                # ดึงข้อความหลัก
                plain_text = json_to_text(course)
                
                # ดึงคำอธิบาย พร้อมดักจับกรณีไม่มีข้อมูล
                try:
                    desc = json_to_desc(course)
                    # หากฟังก์ชันคืนค่าสตริงว่างมา ให้ปรับเป็น None 
                    if not desc: 
                        desc = None
                except Exception:
                    # หากเกิด Error (เช่น ไม่มี Key ใน JSON) ให้ตั้งเป็น None
                    desc = None
                
                # 2. จับคู่ข้อมูลใส่ Dictionary
                data_list.append({
                    'id': id_,
                    'text':plain_text,
                    'is_description':False,
                })
                data_list.append({
                    'id': id_,
                    'text':desc,
                    'is_description':True
                })
                id_+=1
    # 3. แปลง List ของ Dictionary เป็น Pandas DataFrame
    df = pd.DataFrame(data_list)

    return df


def extract_course_and_meta(path = your_path):
    file_path = path
    file_name = os.listdir(file_path)

    # ใช้ List เก็บข้อมูลแทนการ concat ในลูป (ทำงานเร็วกว่ามาก)
    course_list = []
    meta_list = []

    global_course_id = 1
    
    for file in file_name:
        if 'full' not in os.path.basename(file):
            continue
            
        with open(os.path.join(file_path, file), encoding='utf-8', mode='r') as f:
            json_data = json.loads(f.read()).get('courses', [])
            
            for course_dict in json_data:
                # 1. จัดการ Meta Data ก่อน
                if 'source_provenance' in course_dict:
                    for meta in course_dict['source_provenance']:
                        # เพิ่ม foreign key เข้าไปในทุกๆ dictionary ของ meta_data
                        meta['fk_Course'] = global_course_id
                        meta_list.append(meta)
                
                # 2. จัดการ Course Data (แยกข้อมูลที่ไม่ใช่ source_provenance ออกมา)
                course_info = {k: v for k, v in course_dict.items() if k != 'source_provenance'}
                course_info['id'] = global_course_id # แปะ ID ให้ตรงกัน
                course_list.append(course_info)
                
                global_course_id += 1
                
    # นำ List มาแปลงเป็น DataFrame รวดเดียวตอนจบ
    df_course = pd.DataFrame(course_list)
    df_meta = pd.DataFrame(meta_list)
    
    return df_course, df_meta

def vector():
    '''use output from max_text()'''
    vector = vector_model()
    max_text_ = max_text()
    df = max_text_[max_text_['text'].notna()][['text']]
    embeding = vector.word_to_vec(df['text'].to_list())
    df['embed'] = list(embeding)
    df['embed_byte'] = df['embed'].apply(lambda x: np.array(x, dtype=np.float32).tobytes())
    vector_df = pd.merge(df, max_text_, left_index=True, right_index=True, how='right', suffixes=('_df', '_max_text'))
    vector_df.drop(columns=['text_max_text'],inplace=True)
    return vector_df , max_text_

def final():
    '''แก้ชื่อเล็กน้อย
    vector_ , max_text_ , course_ , meta_'''
    vector_df = vector() 
    # 1. เติม .copy() เพื่อป้องกัน Pandas แจ้งเตือน SettingWithCopyWarning
    vector_ = vector_df[0][['embed_byte' , 'is_description' , 'id']].copy()
    vector_.rename(columns={'id':'fk_Course'} , inplace = True)

    # 2. ระบุ subset ให้ชัดเจน ป้องกันการเผลอลบแถวที่มีข้อมูลครบแต่ขาดแค่ฟิลด์อื่น
    vector_.dropna(subset=['embed_byte'], inplace = True)

    max_text_ = vector_df[1].rename(columns={'id':'fk_Course'})
    course_and_meta = extract_course_and_meta()
    course_ = course_and_meta[0]
    meta_ = course_and_meta[1]
    return vector_ , max_text_ , course_ , meta_

