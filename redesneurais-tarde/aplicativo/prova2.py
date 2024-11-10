import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# Inicializa o mixer de áudio
pygame.mixer.init()

# Nome do áudio a ser tocado quando a boca estiver aberta
som_boca_aberta = "nao_vai.mp3"

# Pontos dos olhos e boca
p_olho_esq = [33, 160, 158, 153, 144, 145]  # Índices do olho esquerdo
p_olho_dir = [362, 263, 373, 380, 385, 386]  # Índices do olho direito
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]  # Índices da boca

# Função para calcular o EAR (Eye Aspect Ratio)
def calculo_ear(face, p_olho_esq, p_olho_dir):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        # Olho esquerdo
        olho_esq = face[p_olho_esq]
        # Olho direito
        olho_dir = face[p_olho_dir]

        # EAR para o olho esquerdo
        ear_esq = (np.linalg.norm(olho_esq[1] - olho_esq[5]) + np.linalg.norm(olho_esq[2] - olho_esq[4])) / (2 * np.linalg.norm(olho_esq[0] - olho_esq[3]))

        # EAR para o olho direito
        ear_dir = (np.linalg.norm(olho_dir[1] - olho_dir[5]) + np.linalg.norm(olho_dir[2] - olho_dir[4])) / (2 * np.linalg.norm(olho_dir[0] - olho_dir[3]))

        # Média do EAR
        ear = (ear_esq + ear_dir) / 2
    except Exception as e:
        print(f"Erro ao calcular EAR: {e}")
        ear = 0.0
    return ear

# Função para calcular o MAR (Mouth Aspect Ratio)
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        boca = face[p_boca]

        # Distância vertical da boca (do ponto 13 ao ponto 14)
        dist_vertical = np.linalg.norm(boca[2] - boca[3])

        # Distância horizontal (pontos 78 e 308)
        dist_horizontal = np.linalg.norm(boca[6] - boca[7])

        # MAR
        mar = dist_vertical / dist_horizontal
    except Exception as e:
        print(f"Erro ao calcular MAR: {e}")
        mar = 0.0
    return mar

# Limiares
ear_limiar = 0.27  # Limite para os olhos fechados (sonolência)
mar_limiar = 0.1   # Limite para a boca aberta

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Inicializa o modelo do Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Controle de áudio
boca_aberta_audio_tocando = False

# Loop principal
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        # Processamento da imagem
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = facemesh.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if resultados.multi_face_landmarks:
            for face_landmarks in resultados.multi_face_landmarks:
                # Desenhando as landmarks com espessura e cor ajustadas
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1, circle_radius=1),  # Cor cinza claro e pontos menores
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1, circle_radius=1)  # Conexões mais finas
                )

                # Cálculo do EAR e MAR
                face = face_landmarks.landmark
                ear = calculo_ear(face, p_olho_esq, p_olho_dir)
                mar = calculo_mar(face, p_boca)

                # Exibição dos valores EAR e MAR
                cv2.putText(frame_bgr, f"EAR: {round(ear, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f"MAR: {round(mar, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Verifica se a boca está aberta
                if mar >= mar_limiar:
                    cv2.putText(frame_bgr, "Boca aberta", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not boca_aberta_audio_tocando:
                        pygame.mixer.music.load(som_boca_aberta)
                        pygame.mixer.music.play()
                        boca_aberta_audio_tocando = True
                else:
                    if boca_aberta_audio_tocando:
                        pygame.mixer.music.stop()
                        boca_aberta_audio_tocando = False

                # Verifica se os olhos estão fechados (sonolência)
                if ear < ear_limiar:
                    cv2.putText(frame_bgr, "Olhos fechados", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibe o vídeo com as marcações
        cv2.imshow("Camera", frame_bgr)

        # Encerra ao pressionar 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
