import WebcamModule
import LaneDetection

def main():
    img = WebcamModule.getImg()
    curveVal = LaneDetection.getLaneCurve(img, 1)
    
    sen = 1.3
    maxVAL= 0.3
    if curveVal>maxVAL: curveVal = maxVAL
    if curveVal<-maxVAL: curveVal = -maxVAL
    print(curveVal)
    
    if curveVal>0:
        sen = 1.7
        if curveVal<0.05: curveVal = 0
    else:
        if curveVal>-0.08: curveVal = 0
    print(0.20, -curveVal*sen, 0.05)
    
    # cv2.waitKey(1)

if __name__ == '__main__':
    while True:
        main()
