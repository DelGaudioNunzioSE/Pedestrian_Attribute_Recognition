import cv2
import time

def play_video_with_fps(video_path, desired_fps=30, transformations=None, output_path=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore: Impossibile aprire il video.")
        return

    # Calcola il tempo di attesa tra i frame in secondi
    frame_delay = 1.0 / desired_fps

    # Ottieni le dimensioni del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inizializza il VideoWriter se è fornito un output path
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec video, puoi usare anche 'MJPG' o 'MP4V'
        out = cv2.VideoWriter(output_path, fourcc, desired_fps, (width, height))

    while True:
        start_time = time.time()

        # Leggi un frame
        ret, frame = cap.read()

        if not ret:
            print("Fine del video.")
            break

        ###TRANSFORMATIONS##
        # Applica ogni trasformazione nella lista
        if transformations:
            for transform in transformations:
                if callable(transform):  # Controlla che sia una funzione
                    frame = transform(frame)
                else:
                    raise ValueError("trasformation is not a function.")
        ####################

        # Mostra il frame
        cv2.imshow('Video', frame)

        # Salva il frame nel video di output
        if output_path:
            out.write(frame)

        # Aspetta per mantenere il framerate
        elapsed_time = time.time() - start_time
        time_to_wait = frame_delay - elapsed_time

        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Interrompi con il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# Esempio di utilizzo
#video_path = "./src/Tracking/videos/Atrio.mp4"
#play_video_with_fps(video_path, 60)
