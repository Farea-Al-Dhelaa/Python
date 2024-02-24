import cv2
from playsound import playsound

# قم بتحميل مصنف الكاسكيد للكشف عن الحرائق
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

# قم بفتح جهاز التقاط الفيديو (الكاميرا)
cap = cv2.VideoCapture(0)

# قم بتعيين حجم نافذة العرض إلى قيمة أكبر
cv2.namedWindow('Open Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Open Camera', 1000, 800)

while True:
    # قراءة الإطار الحالي من جهاز التقاط الفيديو
    ret, frame = cap.read()

    # تحويل الإطار إلى صورة غراي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # كشف الحرائق في الإطار الحالي باستخدام مصنف الكاسكيد
    fire = fire_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)

    # الاستعراض على كل منطقة حريق تم اكتشافها
    for (x, y, w, h) in fire:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # طباعة رسالة تشير إلى اكتشاف الحريق
        print('تم اكتشاف حريق')

        # تشغيل ملف صوتي
        playsound('audio.mp3') 

    # عرض الإطار مع مناطق الحرائق المكتشفة
    cv2.imshow('Open Camera', frame)

    # التحقق من ضغط مفتاح '1' للخروج من الحلقة
    if cv2.waitKey(1) == ord('1'): 
        break

# إطلاق جهاز التقاط الفيديو وإغلاق نافذة العرض
cap.release()
cv2.destroyAllWindows()