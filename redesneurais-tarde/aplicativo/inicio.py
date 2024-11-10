import cv2

#capturar a camêra 
cap = cv2.VideoCapture(0)
# enquanto a camera estiver aberta 
while cap.isOpened():
    # sucesso é booleana-0 e 1 
    sucesso,frame = cap.read()
    if not sucesso:
        print('ignorando o fram vazio da camera')
        continue
    cv2.imshow('Camera',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
#fecha a captura
cap.release()
cv2.destroyAllWindows()