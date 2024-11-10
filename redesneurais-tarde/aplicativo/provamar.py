import cv2
import mediapipe as mp
import numpy as np
import pygame

# Inicializa o mixer de áudio
pygame.mixer.init()

# Nome do áudio a ser tocado quando a boca estiver aberta
som_boca_aberta = "nao_vai.mp3"  

# Pontos da boca
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]  # Índices dos pontos da boca

# Função para calcular o MAR (Mouth Aspect Ratio)
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        # Pontos da boca
        ponto_boca_superior = face[p_boca[2]]  # Ponto 13
        ponto_boca_inferior = face[p_boca[3]]  # Ponto 14
        ponto_boca_extrema_esq = face[p_boca[6]]  # Ponto 78
        ponto_boca_extrema_dir = face[p_boca[7]]  # Ponto 308

        # Distâncias verticais (distância entre os pontos superior e inferior)
        distancia_vertical = np.linalg.norm(ponto_boca_superior - ponto_boca_inferior)

        # Distância horizontal (distância entre os pontos extremos da boca)
        distancia_horizontal = np.linalg.norm(ponto_boca_extrema_esq - ponto_boca_extrema_dir)

        # Cálculo do MAR
        mar = distancia_vertical / distancia_horizontal
    except Exception as e:
        print(f"Erro ao calcular MAR: {e}")
        mar = 0.0
    return mar

# Limiares
mar_limiar = 0.1  # Limiar para definir se a boca está aberta

# Inicializa a câmera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Estado do som
boca_aberta_audio_tocando = False 

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if saida_facemesh.multi_face_landmarks:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark

                # Cálculo do MAR e exibição
                mar = calculo_mar(face, p_boca)
                cv2.putText(frame, f"MAR: {round(mar, 2)}", (1, 50),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)

                # Se a boca estiver aberta, toque o áudio da boca aberta
                if mar >= mar_limiar:
                    cv2.putText(frame, "Boca aberta", (50, 200),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
                    if not boca_aberta_audio_tocando:
                        pygame.mixer.music.load(som_boca_aberta)  # Toca o áudio específico para boca aberta
                        pygame.mixer.music.play()
                        boca_aberta_audio_tocando = True
                elif mar < mar_limiar:
                    if boca_aberta_audio_tocando:
                        pygame.mixer.music.stop()  # Para o áudio quando a boca está fechada
                        boca_aberta_audio_tocando = False

        # Exibe a imagem processada
        cv2.imshow('Camera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
