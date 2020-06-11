import cv2
import simpleaudio as sa

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
hand_cascade = cv2.CascadeClassifier("hand.xml")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

    for(x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    if len(hands):
        if hands[0][0] < 160 and hands[0][1] < 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/a-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()
        elif hands[0][0] < 320 and hands[0][1] < 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/b-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()
        elif hands[0][0] < 480 and hands[0][1] < 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/c-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()
        elif hands[0][0] < 640 and hands[0][1] < 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/d-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()
        elif hands[0][0] < 210 and hands[0][1] >= 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/e-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()
        elif hands[0][0] < 420 and hands[0][1] >= 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/f-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()
        elif hands[0][0] < 640 and hands[0][1] >= 240:
            wave_obj = sa.WaveObject.from_wave_file('Guitar_chords/g-major.wav')
            play_obj = wave_obj.play()
            play_obj.wait_done()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

