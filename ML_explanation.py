import streamlit as st

st.title("Machine Learning Model Explanation")

st.header("1. Dataset Description")

st.write("""
Dataset ที่ใช้คือ Stroke Prediction Dataset ซึ่งมีเป้าหมายเพื่อทำนายว่า
บุคคลมีความเสี่ยงจะเป็นโรคหลอดเลือดสมอง (stroke) หรือไม่

Target Variable:
- stroke (0 = ไม่เป็น, 1 = เป็น)

Feature สำคัญ:
- age
- hypertension
- heart_disease
- avg_glucose_level
- bmi
- smoking_status
""")

st.header("2. Data Preparation")

st.write("""
ขั้นตอนเตรียมข้อมูลประกอบด้วย:

1. จัดการ Missing Value
   - ตัวเลขเติมด้วย median
   - ข้อมูลประเภทข้อความเติมด้วยค่าที่พบบ่อยที่สุด

2. One-Hot Encoding
   - แปลงข้อมูล categorical ให้เป็นตัวเลข

3. Standard Scaling
   - ปรับสเกลข้อมูลให้อยู่ในช่วงที่เหมาะสม
""")

st.header("3. Machine Learning Models")

st.write("""
โมเดลที่ใช้เป็น Ensemble Learning แบบ Soft Voting
ประกอบด้วย 3 โมเดล:

1. Random Forest
   - ลด Overfitting ของ Decision Tree
   - เหมาะกับข้อมูลเชิงตาราง

2. Gradient Boosting
   - สร้างโมเดลแบบ Boosting
   - เน้นแก้ข้อผิดพลาดของโมเดลก่อนหน้า

3. Logistic Regression
   - โมเดลเชิงเส้น
   - ใช้เป็น baseline และช่วยให้ ensemble สมดุล
""")

st.header("4. Ensemble Strategy")

st.write("""
ใช้ VotingClassifier แบบ soft voting
โดยนำ probability ของทั้ง 3 โมเดลมาเฉลี่ย
แล้วใช้ threshold 0.5 ในการตัดสินผลลัพธ์
""")

st.header("5. Evaluation")

st.write("""
ประเมินผลด้วย:
- Accuracy
- F1-score
- Confusion Matrix

เนื่องจากข้อมูล stroke มีความไม่สมดุล
จึงพิจารณา F1-score ควบคู่กับ Accuracy
""")