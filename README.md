# Iris Ensemble: Bagging vs. Boosting

เปรียบเทียบ Ensemble ที่ใช้ Classifier ต่างตระกูล 5 ตัว (k-NN, Decision Tree, Naive Bayes, SVM, MLP) บน Iris dataset

- **Bagging**: ใช้ bootstrap sample รันทั้ง 5 โมเดลขนานกันด้วย `ThreadPoolExecutor` แล้วหาคำตอบด้วย Majority Voting ที่เขียนด้วย NumPy
- **Boosting**: AdaBoost แบบตามลำดับ ปรับ `sample_weight` ทุกรอบ ส่วนโมเดลที่ไม่รองรับ `sample_weight` ใช้ resampling แทน

## โครงสร้าง

```text
data/iris.csv         Iris dataset จาก Hugging Face (antoniopelusi/iris)
src/dataset.py        โหลด ตรวจสอบ แบ่ง 80/20 แบบ stratified และ scale ข้อมูล
src/models.py         สร้าง Classifier ทั้ง 5 ตัว
src/bagging.py        Bagging แบบขนาน + Majority Voting
src/boosting.py       Boosting แบบตามลำดับ + weighted vote
main.py               รันการทดลองทั้งหมดและบันทึก output/results.json
output/FINAL_REPORT.md  รายงานผลการทดลอง
```

## วิธีรัน

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

โปรแกรมจะพิมพ์ขนาดข้อมูล ค่า e_m และ alpha ในแต่ละรอบของ Boosting และตาราง Accuracy กับเวลา (ค่ามัธยฐานจาก 10 รอบ) แล้วบันทึกผลไว้ที่ `output/results.json`

ผลการทดลองและการวิเคราะห์อยู่ใน [output/FINAL_REPORT.md](output/FINAL_REPORT.md)
