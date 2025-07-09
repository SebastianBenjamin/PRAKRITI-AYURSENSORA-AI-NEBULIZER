import pygame
import os
import random
import time
import threading
import requests
from testValues import PrakrutiData
import cv2

Pfolder='F:/KJSIM/prakruthi/Module-2 H&S/Software/Tests'
def play_song_from_folder(prakruthi):
    folder_path = Pfolder+'/Songs/' + str(prakruthi)
    
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
    
    if not files:
        print(f"No mp3 files found in folder '{folder_path}'.")
        return
    
    song_file = random.choice(files)
    song_path = os.path.join(folder_path, song_file)
    
    pygame.mixer.init()
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
    
    print(f"Now playing: {song_file}")
    
    # Start thread for pause/resume control
    control_thread = threading.Thread(target=pause_control)
    control_thread.daemon = True  
    control_thread.start()

    print()
    prev_min = -1
    while pygame.mixer.music.get_busy():
        pos_ms = pygame.mixer.music.get_pos()
        pos_sec = pos_ms // 1000
        minutes = pos_sec // 60
        seconds = pos_sec % 60
        
        if minutes != prev_min:
            print(f"\nPosition: {minutes:02d}:{seconds:02d} (mm:ss)")
            prev_min = minutes
        
        time.sleep(0.5)
    
    print("Song finished.")

def play_video(video_path):
    folder_path = Pfolder + '/Videos/' + str(video_path)

    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"No video files found in folder '{folder_path}'.")
        return

    selected_video = random.choice(video_files)
    video_file_path = os.path.join(folder_path, selected_video)

    print(f"Now playing video: {video_file_path}")

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    cv2.namedWindow('Video Playback', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video Playback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Playback', frame)

        # Press 'q' to quit playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video finished.")

def pause_control():
    while True:
        user_input = input("Type 'p' to pause, 'r' to resume, 'q' to stop: ").lower()
        if user_input == 'p':
            pygame.mixer.music.pause()
            print("Music paused.")
        elif user_input == 'r':
            pygame.mixer.music.unpause()
            print("Music resumed.")
        elif user_input == 'q':
            pygame.mixer.music.stop()
            print("Music stopped.")
            break
        else:
            print("Invalid input. Use 'p' for pause, 'r' for resume, 'q' for stop.")

if __name__ == "__main__":
    

    url = "http://127.0.0.1:5000"  
    Prakruti="Null"
    
    data = PrakrutiData

    response = requests.post(f"{url}/predict", json=data)
    print(f"Status code: {response.status_code}")
    if response.status_code==200:
        result = response.json()
        
        Prakruti=result.get('prediction', 'Not found')
        song_no=0
        if "Pittaj" == Prakruti: song_no=1
        elif "Vataj" == Prakruti: song_no=2
        elif "Kaphaj" == Prakruti: song_no=3
        elif "Pittaj-Kaphaj" == Prakruti: song_no=4
        elif "Pittaj-Vataj" == Prakruti: song_no=5
        elif "Vataj-Pittaj" == Prakruti: song_no=6
        elif "Kaphaj-Pittaj" == Prakruti: song_no=7
        else : print(f"Invalid Prakruti : {Prakruti}")
        print(f"\n Prakruti : {Prakruti} \n Playing from Folder No : {song_no}" )
        time.sleep(3)
        # song_thread = threading.Thread(target=play_song_from_folder,args=(song_no,))
        # song_thread.daemon = True  
        # song_thread.start()

        video_thread = threading.Thread(target=play_video,args=(song_no,))
        video_thread.daemon = True  
        video_thread.start()
        play_song_from_folder(song_no)
        # play_video(song_no)

