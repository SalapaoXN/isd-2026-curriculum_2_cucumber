import json
from jiwer import cer, wer


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """ฟังก์ชันทำความสะอาดข้อความและจัดการค่า None ให้เป็น String ว่าง"""
    if text is None:
        return ""
    # ลบช่องว่างหัวท้ายและแปลงเป็นตัวพิมพ์เล็ก (เพื่อไม่ให้ case-sensitive ทำ Error พุ่ง)
    return str(text).strip().lower()


def merge_json_key(dicts):
    key = list(dicts.keys())
    if 'page' in key:
        key.remove('page')
    word = []
    for i in key:
        word.append(clean_text(dicts[i]))
    return word

ocr = load_json(r"C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\new\merged_courses.json") #OCR_file
ground_truth_no_coop = load_json(r"C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\GT-20260712T102537Z-2-001 (1)\GT\DSBA\DSBA_academic_plan_no_coop.json") #ground truth
ground_truth_coop = load_json(r"C:\Users\TUF\OneDrive\Desktop\kmitl\ISD\GT-20260712T102537Z-2-001 (1)\GT\DSBA\DSBA_academic_plan_coop.json") #ground truth


def Field_Level(ground_truth):
    dict_code_ocr = {}
    for i in ocr:
        ocr_merged = merge_json_key(i)
        dict_code_ocr[ocr_merged[0]] = ocr_merged

    count = 0

    for i in ground_truth['courses']:
        gt_merged = merge_json_key(i)
        code = gt_merged[0]
        
        compare = dict_code_ocr.get(code)
        
        if compare is not None:
            print(code)
            try:
                # jiwer: (Ground Truth, OCR)
                print('cer', cer(gt_merged, compare))
                print('wer', wer(gt_merged, compare))
            except ValueError as e:
                # ดักจับ Error กรณี List ของเฉลยมีช่องที่เป็น String ว่าง ("")
                print(f"Error: {e} (ข้ามการคำนวณช่องว่าง)")
            print('\n')
        else:
            count += 1
            
    print('จำนวนวิชาที่ Ground truth มี แต่ผล OCR ไม่มี:', count)


def Page_Level(ground_truth, ocr):
    # 1. แปลง Ground Truth ให้อยู่ในรูปแบบ Dictionary เพื่อให้ดึงข้อมูลมาเทียบได้ง่าย
    dict_gt = {}
    for gt_item in ground_truth['courses']:
        gt_merged = merge_json_key(gt_item)
        code = gt_merged[0] # สมมติว่า index 0 คือรหัสวิชา
        dict_gt[code] = gt_merged

    # 2. สร้าง Dictionary สำหรับเก็บผลรวมคะแนนแยกตามเลขหน้า
    page_metrics = {}

    # 3. วนลูปอ่านผล OCR และกระจายคะแนนลงไปตามแต่ละหน้า
    for ocr_item in ocr:
        ocr_merged = merge_json_key(ocr_item)
        code = ocr_merged[0]
        
        # ถ้ามีวิชานี้ใน Ground Truth ให้ทำการคำนวณ
        if code in dict_gt:
            gt_merged = dict_gt[code]
            
            try:
                # แปลง List เป็น String เดียวกัน เพื่อป้องกัน Error จากค่าว่าง ("") ของ jiwer
                gt_str = " ".join([str(x) for x in gt_merged if x])
                ocr_str = " ".join([str(x) for x in ocr_merged if x])
                
                # ข้ามถ้า Ground Truth เป็นค่าว่าง
                if not gt_str.strip():
                    continue
                    
                c_score = cer(gt_str, ocr_str)
                w_score = wer(gt_str, ocr_str)
                
                # กระจายคะแนนนี้ไปให้ทุกหน้าที่วิชานี้ปรากฏอยู่
                pages = ocr_item.get('page', [])
                for page_num in pages:
                    if page_num not in page_metrics:
                        # สร้างกล่องเก็บคะแนนสำหรับหน้าใหม่
                        page_metrics[page_num] = {'cer_sum': 0, 'wer_sum': 0, 'count': 0}
                        
                    page_metrics[page_num]['cer_sum'] += c_score
                    page_metrics[page_num]['wer_sum'] += w_score
                    page_metrics[page_num]['count'] += 1
                    
            except ValueError as e:
                pass # ข้ามกรณีที่คำนวณไม่ได้

    # 4. สรุปผลและหาค่าเฉลี่ยของแต่ละหน้า
    print("="*50)
    print("📊 สรุปผล Evaluate ระดับ Page Level")
    print("="*50)
    
    total_avg_cer = 0
    total_avg_wer = 0
    valid_pages_count = 0
    
    # เรียงลำดับตามเลขหน้าจากน้อยไปมาก
    for page_num in sorted(page_metrics.keys()):
        metrics = page_metrics[page_num]
        count = metrics['count']
        
        if count > 0:
            avg_cer = metrics['cer_sum'] / count
            avg_wer = metrics['wer_sum'] / count
            
            print(f" หน้าที่ {page_num}: มี {count} วิชา | Avg CER: {avg_cer:.4f} | Avg WER: {avg_wer:.4f}")
            
            total_avg_cer += avg_cer
            total_avg_wer += avg_wer
            valid_pages_count += 1
            
    # 5. สรุปค่าเฉลี่ยรวมทุกหน้า (Grand Average)
    if valid_pages_count > 0:
        print("-" * 50)
        print(f"📌 ค่าเฉลี่ยรวมทั้งหมด ({valid_pages_count} หน้า):")
        print(f"Total Average CER: {total_avg_cer / valid_pages_count:.4f}")
        print(f"Total Average WER: {total_avg_wer / valid_pages_count:.4f}")
    else:
        print("ไม่พบข้อมูลหน้าที่สามารถจับคู่ประเมินผลได้")
    print("="*50)


def Category_Level(ground_truth, ocr):
    # 1. ดึง Ground Truth มาทำเป็น Dictionary เพื่อให้ค้นหาด้วย Code ได้ง่าย
    dict_gt = {}
    for gt_item in ground_truth['courses']:
        gt_merged = merge_json_key(gt_item)
        code = str(gt_merged[0]) # รหัสวิชา
        dict_gt[code] = gt_merged

    # 2. สร้างที่เก็บข้อมูลแยกตามหมวดหมู่
    category_metrics = {
        "060": {'cer_sum': 0, 'wer_sum': 0, 'count': 0},
        "906": {'cer_sum': 0, 'wer_sum': 0, 'count': 0},
        "other": {'cer_sum': 0, 'wer_sum': 0, 'count': 0} # เผื่อมีรหัสแปลกปลอมหลุดมา
    }

    # 3. วนลูปตรวจสอบผล OCR
    for ocr_item in ocr:
        ocr_merged = merge_json_key(ocr_item)
        code = str(ocr_merged[0])
        
        # ถ้าพบวิชานี้ใน Ground Truth
        if code in dict_gt:
            gt_merged = dict_gt[code]
            
            try:
                # แปลง List ให้เป็น String ยาวๆ เพื่อเข้าสมการ jiwer
                gt_str = " ".join([str(x) for x in gt_merged if x])
                ocr_str = " ".join([str(x) for x in ocr_merged if x])
                
                # ถ้าเฉลยว่างเปล่าให้ข้ามไปป้องกันโปรแกรมพัง
                if not gt_str.strip():
                    continue
                    
                # คำนวณค่า Error ของวิชานี้
                c_score = cer(gt_str, ocr_str)
                w_score = wer(gt_str, ocr_str)
                
                # เช็กตัวเลข 3 ตัวแรกของรหัสวิชา เพื่อจัดเข้าหมวดหมู่
                if code.startswith("060"):
                    cat = "060"
                elif code.startswith("906"):
                    cat = "906"
                else:
                    cat = "other"
                    
                # บวกคะแนนสะสมและนับจำนวนวิชาในหมวดนั้น
                category_metrics[cat]['cer_sum'] += c_score
                category_metrics[cat]['wer_sum'] += w_score
                category_metrics[cat]['count'] += 1
                
            except ValueError:
                pass # ข้ามถ้าข้อมูลมีปัญหา

    # 4. สรุปผลลัพธ์และคำนวณค่าเฉลี่ย
    print("="*50)
    print(" สรุปผล Evaluate ระดับ Category Level")
    print("="*50)
    
    # แสดงผลแยกทีละหมวดหมู่
    for cat in ["060", "906", "other"]:
        metrics = category_metrics[cat]
        count = metrics['count']
        
        if count > 0:
            avg_cer = metrics['cer_sum'] / count
            avg_wer = metrics['wer_sum'] / count
            
            print(f" หมวดวิชา [ขึ้นต้นด้วย {cat}]: พบ {count} วิชา")
            print(f"   - Average CER: {avg_cer:.4f}")
            print(f"   - Average WER: {avg_wer:.4f}")
            print("-" * 50)
        elif cat != "other":
            print(f"📁 หมวดวิชา [ขึ้นต้นด้วย {cat}]: ไม่มีข้อมูลวิชาที่สามารถประเมินได้")
            print("-" * 50)


print('ground_truth_coop , Category level')
Category_Level(ground_truth_coop , ocr)
print('ground_truth_no_coop , Category level ')
Category_Level(ground_truth_no_coop , ocr)

print('ground_truth_coop , page level')
Page_Level(ground_truth_coop , ocr)
print('ground_truth_no_coop , page level ')
Page_Level(ground_truth_no_coop , ocr)

print('ground_truth_coop , Field level')
Field_Level(ground_truth_coop)
print('ground_truth_no_coop , Field level ')
Field_Level(ground_truth_no_coop)
