# MotionRec: A Unified Deep Framework for Moving Object Recognition

**Authors**: Murari Mandal, Lav Kush Kumar, Mahipal Singh Saran, Santosh Kumar Vipparthi

**Source**: https://openaccess.thecvf.com/content_WACV_2020/papers/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.pdf

---

## Zusammenfassung

* Lokalisierung und Klassifizierung von bewegenden Objekten in Videos
* Unterscheidung zwischen Hintergrund, statischen und sich bewegende Objekte
* Probleme: Schatten, Hintergrundänderungen, ...
* Ähnlich: Moving Object Detection (MOD), Object Detection
* MOD filtert nach Bewegungen, aber nicht nach Klassen
* Object Detection lokalisiert und klassifiziert Objekte in statischen Bildern und berücksichtigt nicht zeitlich abhängige Änderungen, um nur bewegte Objekte zu erkennen
* MOR: Abschätzen des Hintergrundes vergangener Frames mit temporal depth reductionist (TDR)