import cv2, time
from datetime import datetime
import pandas

first_frame=None
status_list=[None, None]
times=[]
df=pandas.DataFrame(columns=["Start", "End"])

video=cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    #učitava boolean check i numpy array za images
    check, frame = video.read()

    #varijabla za motion status
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21, 21), 0)

    #uzima si background frame s kojim će uspoređivati
    if first_frame is None:
        first_frame=gray
        continue

    #razliku između prvog frame-a i gray spremamo u deltaframe
    delta_frame=cv2.absdiff(first_frame,gray)

    #dodijeljuje onome u numpy polju što se razlikuje između ovih slika za više od 30 bijelu boju
    #to je tuple pa nam treba samo ono na [1] mjestu
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    #smoothanje onih crnih rupa u tresholdu, removanje tih crnih rupa u 2 iteracije (iterations)
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #contour detection frame koji unesemo
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #ako je područje countoura manje od 1000 pixela, nastavi
    #ako ne napravi rectangle oko tog područja
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1
        #ovdje dobijemo 4 koordinate pa s njima kasnije crtamo rectangle
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    #time.sleep(3)
    #dodajemo status u listu pa ćemo promjene iz 0 u 1 i 1 u 0 iz liste bilježiti vremena
    status_list.append(status)
    #trebaju nam uvijek samo zadnja dva itema, pa impreoveamo zbog memorije
    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())

    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("Prozor", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
    

print(times)
print(status_list)

#punimo pandas datafrime s timestampsima od datetime.now    
for i in range(0, len(times), 2):
    df=df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows
